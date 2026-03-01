use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

pub struct RawChunk {
    pub name: String,
    pub source: String,
    pub embed_text: String,
    pub kind: String,
}

struct LangConfig {
    language: Language,
    chunk_query: &'static str,
}

fn lang_for_ext(ext: &str) -> Option<LangConfig> {
    match ext {
        "rs" => Some(LangConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            chunk_query: r#"
                (function_item name: (identifier) @name) @chunk
                (struct_item name: (type_identifier) @name) @chunk
                (enum_item name: (type_identifier) @name) @chunk
                (trait_item name: (type_identifier) @name) @chunk
                (impl_item) @chunk
            "#,
        }),
        "py" => Some(LangConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            chunk_query: r#"
                (function_definition name: (identifier) @name) @chunk
                (class_definition name: (identifier) @name) @chunk
            "#,
        }),
        _ => None,
    }
}

pub fn parse_file(path: &Path, source: &str) -> Vec<RawChunk> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let config = match lang_for_ext(ext) {
        Some(c) => c,
        None => return vec![],
    };

    let mut parser = Parser::new();
    parser
        .set_language(&config.language)
        .expect("tree-sitter language load must not fail");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => {
            eprintln!("tree-sitter: parse failed for {}", path.display());
            return vec![];
        }
    };

    let query = match Query::new(&config.language, config.chunk_query) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("tree-sitter: invalid query for {ext}: {e:?}");
            return vec![];
        }
    };

    let chunk_idx = match query.capture_index_for_name("chunk") {
        Some(i) => i,
        None => {
            eprintln!("tree-sitter: query missing @chunk capture for {ext}");
            return vec![];
        }
    };
    let name_idx = query.capture_index_for_name("name");

    let mut cursor = QueryCursor::new();
    let mut chunks = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    while let Some(m) = matches.next() {
        let chunk_cap = match m.captures.iter().find(|c| c.index == chunk_idx) {
            Some(c) => c,
            None => continue,
        };
        let name_cap = name_idx.and_then(|ni| m.captures.iter().find(|c| c.index == ni));

        let node = chunk_cap.node;
        let source_text = &source[node.byte_range()];
        let embed_text = if source_text.len() > 2000 {
            &source_text[..2000]
        } else {
            source_text
        };
        let name = name_cap
            .map(|n| source[n.node.byte_range()].to_string())
            .unwrap_or_else(|| "(anonymous)".to_string());

        chunks.push(RawChunk {
            name,
            source: source_text.to_string(),
            embed_text: embed_text.to_string(),
            kind: node.kind().to_string(),
        });
    }

    chunks
}
