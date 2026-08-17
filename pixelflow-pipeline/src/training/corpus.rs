//! Binary corpus format for pre-parsed expression storage.
//!
//! Replaces JSONL text corpus with a binary format that loads in microseconds
//! via sequential read (no parsing, no allocation beyond the arena vecs).
//!
//! ## Format (v3)
//!
//! ```text
//! magic: [u8; 4] = b"PXCR"
//! version: u32 (little-endian) = 3
//! count: u32 (little-endian)
//!
//! For each expression:
//!   name_len: u16 (little-endian)
//!   name: [u8; name_len]       (UTF-8)
//!   node_count: u32 (le)
//!   nary_count: u32 (le)
//!   root_index: u32 (le)       (ExprId.0)
//!   nodes: node_count encoded ExprNodes (variable per-node)
//!   nary_children: [u32; nary_count] (le) (ExprId.0 values)
//! ```
//!
//! Each ExprNode is encoded as:
//!   tag: u8  (0=Var, 1=Const, 2=Param, 3=Unary, 4=Binary, 5=Ternary, 6=Nary)
//!   payload varies by tag.
//!
//! ## Only the reachable subtree is stored
//!
//! [`write_corpus`] serializes [`reachable_subtree`], not the caller's whole
//! arena. Generator arenas are append-only scratch space: rewriting a node
//! pushes a replacement and abandons the original, so an arena holding a
//! 12-node expression can carry hundreds of dead nodes. Storing them made
//! `arena.len()` a measure of the *generator's history* rather than of the
//! expression, and every downstream size filter that read `arena.len()`
//! silently dropped small expressions with long provenance — non-randomly,
//! since dead-node count correlates with how many rewrite passes ran (the
//! B3 holdout-integrity bug: 110 of 380 DEV entries dropped by a filter
//! measuring dead nodes). After compaction, stored size *is* expression size.
//!
//! ## Version is coupled to the payload's meaning
//!
//! Two independent things invalidate a stored corpus, and neither announces
//! itself as a parse error. `VERSION` is what stands between both of them and
//! a silent misread, so any version other than the current one is a hard load
//! error.
//!
//! **The op encoding can change under us.** Ops are stored in `pixelflow-ir`'s
//! own encoding (`OpKind::marshal`), whose bytes that crate is free to change
//! without telling anyone. A corpus written under a different encoding parses
//! perfectly and decodes into different operations, so changing the encoding
//! means bumping the version here. That is what v2 marked — the dense 0..COUNT
//! opcode renumbering.
//!
//! **The payload's meaning can change while its bytes stay well-formed.** v3
//! marks reachable-subtree compaction: a v2 file parses perfectly, but its node
//! counts include the generator's dead nodes and so mean something else
//! entirely. Both bumps guard the same silently-wrong-data shape from opposite
//! directions.

use std::io::{self, Write};
use std::path::Path;

use pixelflow_ir::kind::OpCode;
use pixelflow_ir::{ExprArena, ExprId, ExprNode, OpKind};

const MAGIC: &[u8; 4] = b"PXCR";
// Bump this whenever anything about the bytes below changes meaning — the
// header's shape, a node tag, the encoding `OpKind::marshal` writes ops in, or
// which nodes the payload carries.
// See `docs/designs/opkind-numbering-is-private.md` §4.4.
// The encoding one is easy to forget precisely because it changes nothing
// here: the file still parses, and every op byte quietly names a different
// operation. A stale corpus is cheap to replace and expensive to misread, so
// when in doubt, bump.
//
// 1 → 2: OpKind discriminants were renumbered dense (0..COUNT); ops serialize
//        through `OpKind::marshal`, so a v1 corpus decodes its op bytes as
//        different ops under the new numbering.
// 2 → 3: entries now store only the subtree reachable from the root. A v2
//        file still *parses*, which is worse than a parse error — its node
//        counts include the generator's dead nodes, so every size filter
//        reading a v2 corpus measures arena history instead of expression
//        size (bug B3).
// Both bumps turn silent corruption into a load-time error.
const VERSION: u32 = 3;

// ── ExprNode serialization tags ──────────────────────────────────────────────

const TAG_VAR: u8 = 0;
const TAG_CONST: u8 = 1;
const TAG_PARAM: u8 = 2;
const TAG_UNARY: u8 = 3;
const TAG_BINARY: u8 = 4;
const TAG_TERNARY: u8 = 5;
const TAG_NARY: u8 = 6;
const TAG_BUFFER: u8 = 7;

// ── Reachable-subtree compaction ─────────────────────────────────────────────

/// Rebuild `(arena, root)` keeping only the nodes reachable from `root`.
///
/// DAG sharing is preserved: a node referenced from several parents is
/// emitted once and referenced by the same new id, so compaction never
/// expands a shared subgraph into a tree. Children are emitted before their
/// parents, so the result satisfies [`ExprArena::from_raw`]'s ordering
/// contract.
///
/// This is what makes stored size mean expression size. Callers that
/// generate expressions into a long-lived scratch arena (`BwdGenerator`, the
/// junkify passes, e-graph extraction) leave dead nodes behind by design —
/// an append-only arena has no other way to rewrite — and `arena.len()`
/// counts those. Any consumer that treats `arena.len()` as "how big is this
/// expression" is measuring the wrong population.
///
/// The buffer table is NOT carried over: the corpus format has never
/// serialized buffer declarations, and this function exists to serve it.
/// Compacting an arena with `Buffer` leaves would therefore produce an entry
/// whose declarations were already going to be lost, so it is refused here
/// rather than written out silently broken.
///
/// # Panics
///
/// Panics if the reachable subgraph contains a `Buffer` node.
#[must_use]
pub fn reachable_subtree(arena: &ExprArena, root: ExprId) -> (ExprArena, ExprId) {
    enum Task {
        Descend(ExprId),
        Emit(ExprId),
    }

    let mut id_map: Vec<Option<ExprId>> = vec![None; arena.len()];
    let mut nodes: Vec<ExprNode> = Vec::new();
    let mut nary_children: Vec<ExprId> = Vec::new();
    let mut work: Vec<Task> = vec![Task::Descend(root)];

    while let Some(task) = work.pop() {
        match task {
            Task::Descend(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                work.push(Task::Emit(id));
                let children: Vec<ExprId> = arena.children(id).collect();
                for child in children.into_iter().rev() {
                    work.push(Task::Descend(child));
                }
            }
            Task::Emit(id) => {
                if id_map[id.0 as usize].is_some() {
                    continue;
                }
                let map = |old: ExprId| -> ExprId {
                    id_map[old.0 as usize]
                        .expect("reachable_subtree: child must be emitted before its parent")
                };
                let compacted = match arena.node(id) {
                    ExprNode::Var(i) => ExprNode::Var(*i),
                    ExprNode::Const(v) => ExprNode::Const(*v),
                    ExprNode::Param(i) => ExprNode::Param(*i),
                    ExprNode::Buffer(b) => panic!(
                        "reachable_subtree: expression references Buffer({}), whose declaration \
                         the corpus format does not serialize — writing it would store a node \
                         that cannot be read back correctly",
                        b.0
                    ),
                    ExprNode::Unary(op, a) => ExprNode::Unary(*op, map(*a)),
                    ExprNode::Binary(op, a, b) => ExprNode::Binary(*op, map(*a), map(*b)),
                    ExprNode::Ternary(op, a, b, c) => {
                        ExprNode::Ternary(*op, map(*a), map(*b), map(*c))
                    }
                    ExprNode::Nary(op, start, len) => {
                        let start_new = nary_children.len() as u32;
                        for child in arena.nary_children_slice(*start, *len) {
                            nary_children.push(map(*child));
                        }
                        ExprNode::Nary(*op, start_new, *len)
                    }
                };
                let new_id = ExprId(nodes.len() as u32);
                nodes.push(compacted);
                id_map[id.0 as usize] = Some(new_id);
            }
        }
    }

    let new_root = id_map[root.0 as usize].expect("reachable_subtree: root was never emitted");
    (ExprArena::from_raw(nodes, nary_children), new_root)
}

// ── Write ────────────────────────────────────────────────────────────────────

/// Write a binary corpus to `path`.
///
/// Each entry is compacted with [`reachable_subtree`] before serialization,
/// so the stored node count is the expression's size, not its arena's.
///
/// # Panics
///
/// Panics if any expression name exceeds `u16::MAX` bytes, or if any
/// expression references a `Buffer` node (see [`reachable_subtree`]).
pub fn write_corpus(path: &Path, entries: &[(String, ExprArena, ExprId)]) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = io::BufWriter::new(file);

    // Header
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;

    for (name, arena, root) in entries {
        write_entry(&mut w, name, arena, *root)?;
    }

    w.flush()?;
    Ok(())
}

fn write_entry(w: &mut impl Write, name: &str, arena: &ExprArena, root: ExprId) -> io::Result<()> {
    let name_bytes = name.as_bytes();
    assert!(
        name_bytes.len() <= u16::MAX as usize,
        "write_corpus: expression name exceeds u16::MAX bytes: '{}'",
        name
    );

    // Compact first: the caller's arena is generator scratch space and its
    // dead nodes are not part of this expression (format v3).
    let (compact, compact_root) = reachable_subtree(arena, root);
    let nodes = compact.nodes_raw();
    let nary = compact.nary_children_raw();

    w.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
    w.write_all(name_bytes)?;
    w.write_all(&(nodes.len() as u32).to_le_bytes())?;
    w.write_all(&(nary.len() as u32).to_le_bytes())?;
    w.write_all(&compact_root.0.to_le_bytes())?;

    // Nodes
    for node in nodes {
        write_node(w, node)?;
    }

    // Nary children
    for child in nary {
        w.write_all(&child.0.to_le_bytes())?;
    }

    Ok(())
}

fn write_node(w: &mut impl Write, node: &ExprNode) -> io::Result<()> {
    match node {
        ExprNode::Var(i) => {
            w.write_all(&[TAG_VAR, *i])?;
        }
        ExprNode::Const(v) => {
            w.write_all(&[TAG_CONST])?;
            w.write_all(&v.to_le_bytes())?;
        }
        ExprNode::Param(i) => {
            w.write_all(&[TAG_PARAM, *i])?;
        }
        ExprNode::Unary(op, a) => {
            w.write_all(&[TAG_UNARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
        }
        ExprNode::Binary(op, a, b) => {
            w.write_all(&[TAG_BINARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
        }
        ExprNode::Ternary(op, a, b, c) => {
            w.write_all(&[TAG_TERNARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&a.0.to_le_bytes())?;
            w.write_all(&b.0.to_le_bytes())?;
            w.write_all(&c.0.to_le_bytes())?;
        }
        ExprNode::Nary(op, start, len) => {
            w.write_all(&[TAG_NARY])?;
            w.write_all(&op.marshal().to_bytes())?;
            w.write_all(&start.to_le_bytes())?;
            w.write_all(&len.to_le_bytes())?;
        }
        ExprNode::Buffer(b) => {
            w.write_all(&[TAG_BUFFER])?;
            w.write_all(&b.0.to_le_bytes())?;
        }
    }
    Ok(())
}

// ── Read ─────────────────────────────────────────────────────────────────────

/// Read a binary corpus from `path`.
///
/// Returns `(name, arena, root)` triples.
pub fn read_corpus(path: &Path) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
    let data = std::fs::read(path)?;
    read_corpus_bytes(&data)
}

fn read_corpus_bytes(data: &[u8]) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
    let mut r = Cursor::new(data);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad corpus magic: expected {:?}, got {:?}", MAGIC, magic),
        ));
    }

    // Exact-version check, no tolerance. Op bytes mean whatever the IR's
    // encoding said when the file was written, and that encoding may change
    // without notice — so a "best effort" read of an old file would decode
    // valid-looking bytes into the wrong ops rather than fail. Both prior
    // changes are silent-corruption shaped rather than parse-error shaped:
    // v1's op bytes decode as different ops under the dense OpKind numbering,
    // and v2's node counts include generator dead nodes, so v2 size filters
    // measure the wrong thing. Neither would fail to parse.
    let version = r.read_u32()?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported corpus version: {version} (expected {VERSION}); the OpKind \
                 opcode numbering changed (v1→v2) and entries are now stored as the \
                 reachable subtree only (v2→v3), so older corpora decode to the wrong ops \
                 or the wrong node counts — regenerate the corpus with `cargo run --release \
                 -p pixelflow-pipeline --features training --bin gen_bench_corpus`"
            ),
        ));
    }

    let count = r.read_u32()? as usize;
    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let entry = read_entry(&mut r)
            .map_err(|e| io::Error::new(e.kind(), format!("corpus entry {i}/{count}: {e}")))?;
        entries.push(entry);
    }

    Ok(entries)
}

fn read_entry(r: &mut Cursor<'_>) -> io::Result<(String, ExprArena, ExprId)> {
    let name_len = r.read_u16()? as usize;
    let name = {
        let bytes = r.read_bytes(name_len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid UTF-8 name: {e}"),
            )
        })?
    };

    let node_count = r.read_u32()? as usize;
    let nary_count = r.read_u32()? as usize;
    let root_index = r.read_u32()?;

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(read_node(r)?);
    }

    let mut nary_children = Vec::with_capacity(nary_count);
    for _ in 0..nary_count {
        nary_children.push(ExprId(r.read_u32()?));
    }

    let arena = ExprArena::from_raw(nodes, nary_children);
    let root = ExprId(root_index);

    Ok((name, arena, root))
}

fn read_node(r: &mut Cursor<'_>) -> io::Result<ExprNode> {
    let tag = r.read_u8()?;
    match tag {
        TAG_VAR => {
            let i = r.read_u8()?;
            Ok(ExprNode::Var(i))
        }
        TAG_CONST => {
            let bits = r.read_u32()?;
            Ok(ExprNode::Const(f32::from_le_bytes(bits.to_le_bytes())))
        }
        TAG_PARAM => {
            let i = r.read_u8()?;
            Ok(ExprNode::Param(i))
        }
        TAG_UNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            Ok(ExprNode::Unary(op, a))
        }
        TAG_BINARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            Ok(ExprNode::Binary(op, a, b))
        }
        TAG_TERNARY => {
            let op = read_opkind(r)?;
            let a = ExprId(r.read_u32()?);
            let b = ExprId(r.read_u32()?);
            let c = ExprId(r.read_u32()?);
            Ok(ExprNode::Ternary(op, a, b, c))
        }
        TAG_NARY => {
            let op = read_opkind(r)?;
            let start = r.read_u32()?;
            let len = r.read_u16()?;
            Ok(ExprNode::Nary(op, start, len))
        }
        TAG_BUFFER => {
            let b = pixelflow_ir::arena::BufferId(r.read_u16()?);
            Ok(ExprNode::Buffer(b))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown ExprNode tag: {tag}"),
        )),
    }
}

fn read_opkind(r: &mut Cursor<'_>) -> io::Result<OpKind> {
    let mut bytes = [0u8; OpCode::SIZE];
    for b in &mut bytes {
        *b = r.read_u8()?;
    }
    OpKind::unmarshal(OpCode::from_bytes(bytes)).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no op is encoded by {bytes:?}"),
        )
    })
}

// ── Minimal cursor for zero-copy reads ───────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let end = self.pos + buf.len();
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_exact: need {} bytes at offset {}, but only {} remain",
                    buf.len(),
                    self.pos,
                    self.data.len() - self.pos
                ),
            ));
        }
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_bytes: need {n} bytes at offset {}, but only {} remain",
                    self.pos,
                    self.data.len() - self.pos
                ),
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read_u8: at offset {}, no bytes remain", self.pos),
            ));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Per-process path: concurrent `cargo test` runs must not share corpus files,
    // or one process's remove_file races another's write/read.
    fn unique_tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("corpus_rt_{name}_{}.bin", std::process::id()))
    }

    #[test]
    fn round_trip_empty() {
        let tmp = unique_tmp("empty");
        let entries: Vec<(String, ExprArena, ExprId)> = Vec::new();
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_simple() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let root = arena.push_binary(OpKind::Add, x, y);

        let entries = vec![("test_add".to_string(), arena, root)];

        let tmp = unique_tmp("simple");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "test_add");
        assert_eq!(loaded[0].1.len(), 3);
        assert_eq!(loaded[0].2.0, root.0);

        // Verify node equality
        for (i, node) in entries[0].1.nodes_raw().iter().enumerate() {
            assert_eq!(
                node,
                loaded[0].1.node(ExprId(i as u32)),
                "node {i} mismatch"
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_with_const_and_unary() {
        let mut arena = ExprArena::new();
        let c = arena.push_const(std::f32::consts::PI);
        let root = arena.push_unary(OpKind::Sqrt, c);

        let entries = vec![("sqrt_pi".to_string(), arena, root)];

        let tmp = unique_tmp("unary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "sqrt_pi");
        // Check the const value round-trips
        match loaded[0].1.node(ExprId(0)) {
            ExprNode::Const(v) => assert!(
                (v - std::f32::consts::PI).abs() < 1e-6,
                "const mismatch: {v}"
            ),
            other => panic!("expected Const, got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_ternary() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let root = arena.push_ternary(OpKind::Select, x, y, z);

        let entries = vec![("select_xyz".to_string(), arena, root)];

        let tmp = unique_tmp("ternary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        match loaded[0].1.node(loaded[0].2) {
            ExprNode::Ternary(OpKind::Select, _, _, _) => {}
            other => panic!("expected Ternary(Select,...), got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn round_trip_nary() {
        let mut arena = ExprArena::new();
        let a = arena.push_var(0);
        let b = arena.push_var(1);
        let c = arena.push_var(2);
        let root = arena.push_nary(OpKind::Tuple, &[a, b, c]);

        let entries = vec![("tuple_abc".to_string(), arena, root)];

        let tmp = unique_tmp("nary");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 1);
        match loaded[0].1.node(loaded[0].2) {
            ExprNode::Nary(OpKind::Tuple, start, len) => {
                assert_eq!(*len, 3);
                let children = loaded[0].1.nary_children_slice(*start, *len);
                assert_eq!(children.len(), 3);
            }
            other => panic!("expected Nary(Tuple,...), got {other:?}"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    // Header-rejection tests drive the reader through its public entry point:
    // `read_corpus_bytes` is private, and pinning it here would test a path no
    // caller can reach. `name` keeps sibling tests off each other's fixture
    // within a process, `unique_tmp` keeps concurrent test processes apart.
    fn read_corpus_from_bytes(
        name: &str,
        data: &[u8],
    ) -> io::Result<Vec<(String, ExprArena, ExprId)>> {
        let tmp = unique_tmp(name);
        std::fs::write(&tmp, data).expect("write fixture");
        let result = read_corpus(&tmp);
        let _ = std::fs::remove_file(&tmp);
        result
    }

    #[test]
    fn bad_magic_fails() {
        let data = b"BADMxxxxxxxx";
        match read_corpus_from_bytes("bad_magic", data) {
            Ok(_) => panic!("expected error for bad magic"),
            Err(e) => assert!(
                e.to_string().contains("bad corpus magic"),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn bad_version_fails() {
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // bad version
        data.extend_from_slice(&0u32.to_le_bytes()); // count=0
        match read_corpus_from_bytes("bad_version", &data) {
            Ok(_) => panic!("expected error for bad version"),
            Err(e) => assert!(
                e.to_string().contains("unsupported corpus version"),
                "unexpected error: {e}"
            ),
        }
    }

    #[test]
    fn v1_corpus_is_refused_with_regeneration_hint() {
        // A v1 header must be a hard error: it was written under a different
        // op encoding, so its op bytes decode as different ops — the reader
        // must refuse it, not fall back to decoding garbage.
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes()); // written under an older encoding
        data.extend_from_slice(&0u32.to_le_bytes()); // count=0
        match read_corpus_from_bytes("v1_refused", &data) {
            Ok(_) => panic!("v1 corpus must be refused: its op bytes decode as wrong OpKinds"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("unsupported corpus version: 1"),
                    "error must name the rejected version: {msg}"
                );
                assert!(
                    msg.contains("opcode numbering changed"),
                    "error must explain WHY v1 is rejected: {msg}"
                );
                assert!(
                    msg.contains("gen_bench_corpus"),
                    "error must name the regeneration binary: {msg}"
                );
            }
        }
    }

    #[test]
    fn v2_corpus_is_refused_with_regeneration_hint() {
        // v2 parses byte-for-byte, which is precisely why it must be refused:
        // its entries carry the generator's dead nodes, so a v2 file read by
        // v3 code reports arena history as expression size and every
        // node-count filter silently drops the wrong expressions (bug B3).
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes()); // pre-compaction version
        data.extend_from_slice(&0u32.to_le_bytes()); // count=0
        match read_corpus_from_bytes("v2_refused", &data) {
            Ok(_) => panic!("v2 corpus must be refused: its node counts include dead nodes"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("unsupported corpus version: 2"),
                    "error must name the rejected version: {msg}"
                );
                assert!(
                    msg.contains("reachable subtree"),
                    "error must explain WHY v2 is rejected: {msg}"
                );
                assert!(
                    msg.contains("gen_bench_corpus"),
                    "error must name the regeneration binary: {msg}"
                );
            }
        }
    }

    // ── Reachable-subtree compaction (bug B3) ───────────────────────────────

    /// An arena carrying the dead nodes an append-only generator leaves
    /// behind: `X + 1.0` is built, then abandoned in favour of `X * 2.0`.
    fn arena_with_dead_nodes() -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let _abandoned = arena.push_binary(OpKind::Add, x, one);
        let two = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, x, two);
        (arena, root)
    }

    #[test]
    fn compaction_drops_dead_nodes_and_preserves_the_expression() {
        let (arena, root) = arena_with_dead_nodes();
        assert_eq!(arena.len(), 5, "fixture should carry 2 dead nodes");
        assert_eq!(arena.node_count_subtree(root), 3);

        let (compact, compact_root) = reachable_subtree(&arena, root);
        assert_eq!(compact.len(), 3, "only the reachable DAG survives");
        assert_eq!(compact.node_count_subtree(compact_root), 3);
        assert!(
            compact.subtree_eq(compact_root, &arena, root),
            "compaction must preserve the expression, not just its size"
        );
    }

    #[test]
    fn compaction_preserves_dag_sharing() {
        // `s + s` where `s = sqrt(X)`: the shared child must stay shared, or
        // compaction would turn a DAG into an exponentially larger tree.
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let s = arena.push_unary(OpKind::Sqrt, x);
        let root = arena.push_binary(OpKind::Add, s, s);

        let (compact, compact_root) = reachable_subtree(&arena, root);
        assert_eq!(compact.len(), 3, "shared node must be emitted once");
        match compact.node(compact_root) {
            ExprNode::Binary(OpKind::Add, a, b) => {
                assert_eq!(a, b, "both operands must reference the same node");
            }
            other => panic!("expected Binary(Add, ..), got {other:?}"),
        }
    }

    #[test]
    fn stored_size_reflects_the_expression_not_the_arena() {
        // The B3 regression: a small expression in a junk-heavy arena must
        // round-trip as a small expression. Reading back `arena.len() == 5`
        // is what let a `> N` filter drop 29% of the DEV tier.
        let (arena, root) = arena_with_dead_nodes();
        let entries = vec![("junky".to_string(), arena.clone(), root)];

        let tmp = unique_tmp("dead_nodes");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.len(), 1);
        let (_, loaded_arena, loaded_root) = &loaded[0];
        assert_eq!(
            loaded_arena.len(),
            3,
            "stored arena must hold only the reachable subtree, got {} nodes",
            loaded_arena.len()
        );
        assert_eq!(loaded_arena.node_count_subtree(*loaded_root), 3);
        assert!(
            loaded_arena.subtree_eq(*loaded_root, &arena, root),
            "the round-tripped expression must equal the original"
        );
    }

    #[test]
    #[should_panic(expected = "does not serialize")]
    fn compaction_refuses_buffer_nodes() {
        // A Buffer leaf's declaration is not part of the corpus format, so a
        // corpus entry holding one is unreadable-by-construction. Refuse at
        // write time rather than storing a node that decodes to nothing.
        let nodes = vec![ExprNode::Buffer(pixelflow_ir::arena::BufferId(0))];
        let arena = ExprArena::from_raw(nodes, Vec::new());
        let _ = reachable_subtree(&arena, ExprId(0));
    }

    #[test]
    fn round_trip_multiple_entries() {
        let mut entries = Vec::new();

        // Entry 1: X + Y
        let mut a1 = ExprArena::new();
        let x = a1.push_var(0);
        let y = a1.push_var(1);
        let r1 = a1.push_binary(OpKind::Add, x, y);
        entries.push(("add_xy".to_string(), a1, r1));

        // Entry 2: sqrt(pi)
        let mut a2 = ExprArena::new();
        let c = a2.push_const(std::f32::consts::PI);
        let r2 = a2.push_unary(OpKind::Sqrt, c);
        entries.push(("sqrt_pi".to_string(), a2, r2));

        let tmp = unique_tmp("multi");
        write_corpus(&tmp, &entries).expect("write");
        let loaded = read_corpus(&tmp).expect("read");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "add_xy");
        assert_eq!(loaded[1].0, "sqrt_pi");
        assert_eq!(loaded[0].1.len(), 3);
        assert_eq!(loaded[1].1.len(), 2);

        let _ = std::fs::remove_file(&tmp);
    }
}
