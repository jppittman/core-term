//! The one research-journal writer.
//!
//! A journal line is a claim about a run: what config produced it, what
//! numbers came out. Two binaries write one — `bootstrap_extraction_head`
//! (training quality) and `bench_extraction_3way` (extraction-policy
//! benchmarks) — each with its own record schema (the fields genuinely
//! differ; that stays), but both need the *same* append mechanics: serialize
//! to one JSON line, create the parent directory if it's missing, refuse to
//! corrupt an unsmudged Git LFS pointer, append-or-panic. That mechanism was
//! two copies (one of them missing the LFS guard) before this module
//! (docs/plans/2026-08-17-cost-model-domain.md, J15) — a fix to one could
//! silently fail to reach the other, the same NO-SILENT-FAILURES class as
//! J14's `run_scalar`.
use std::fs;
use std::io::Write as _;
use std::path::Path;

use serde::Serialize;

/// Refuse to append to a journal that is still a Git LFS pointer.
///
/// `.gitattributes` sends every `*.jsonl` through LFS, so in a clone where the
/// objects were never pulled, the journal path holds the three-line pointer
/// stub instead of the journal. Appending there produces a file that is
/// neither — malformed JSONL behind a pointer header — and staging it can
/// commit that text as the tracked payload, taking the real history with it.
/// A run that got this far has already spent its measurements, so say
/// exactly what to run rather than failing in a way that reads as a corrupt
/// journal.
fn refuse_unsmudged_lfs_pointer(path: &Path) {
    const POINTER_MAGIC: &str = "version https://git-lfs.github.com/spec/";
    let Ok(head) = fs::read(path) else {
        return; // Absent is fine — the append creates it.
    };
    // A pointer stub is a few hundred bytes; a real journal's first line is a
    // JSON object. Only the prefix needs looking at.
    let prefix = String::from_utf8_lossy(&head[..head.len().min(POINTER_MAGIC.len())]);
    assert!(
        prefix != POINTER_MAGIC,
        "{} is an unsmudged Git LFS pointer, not the journal.\n\
         Appending would corrupt it. Materialize it first:\n  \
         git lfs install && git lfs pull --include {}",
        path.display(),
        path.display()
    );
}

/// Serialize `record` to one JSON line and append it to the journal at
/// `path`, creating the parent directory if needed. Every failure — encode,
/// mkdir, open, write — is a hard panic naming the path: a journal line that
/// silently didn't land is a run whose claim can never be checked again.
pub fn append_record<T: Serialize>(path: &Path, record: &T) {
    let line = serde_json::to_string(record).unwrap_or_else(|e| {
        panic!(
            "failed to serialize journal record for {}: {e}",
            path.display()
        )
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
    }
    refuse_unsmudged_lfs_pointer(path);
    let mut journal = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    writeln!(journal, "{line}")
        .unwrap_or_else(|e| panic!("failed to append to {}: {e}", path.display()));
    eprintln!("Journal appended: {}", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Rec {
        n: u32,
    }

    #[test]
    fn append_record_creates_parent_dir_and_appends_one_line_per_call() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("pf_journal_test_{}", std::process::id()));
        let path = dir.join("nested/journal.jsonl");
        let _ = fs::remove_dir_all(&dir);

        append_record(&path, &Rec { n: 1 });
        append_record(&path, &Rec { n: 2 });

        let contents = fs::read_to_string(&path).expect("journal file should exist");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines, vec!["{\"n\":1}", "{\"n\":2}"]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "unsmudged Git LFS pointer")]
    fn append_record_refuses_an_lfs_pointer_stub() {
        let mut path = std::env::temp_dir();
        path.push(format!("pf_journal_lfs_test_{}.jsonl", std::process::id()));
        fs::write(
            &path,
            "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n",
        )
        .expect("write pointer stub");

        append_record(&path, &Rec { n: 1 });
    }
}
