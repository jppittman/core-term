use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::{cell::RefCell, path::Path};
use tokenizers::Tokenizer;

pub struct Embedder {
    // RefCell because Session::run takes &mut self, but Embedder is shared via &.
    session: RefCell<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        // Ensure model files are present, downloading from HuggingFace if needed.
        crate::download::ensure_model(model_dir)?;

        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| format!("ort session builder failed: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort optimization level failed: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| {
                format!(
                    "failed to load ONNX model from {}: {e}",
                    model_path.display()
                )
            })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            )
        })?;

        Ok(Self {
            session: RefCell::new(session),
            tokenizer,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        // Truncate to 512 tokens max (EmbeddingGemma supports 2K but 512 is sufficient
        // for code chunks and keeps inference fast)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let seq_len = ids.len().min(512);
        let ids = ids[..seq_len].to_vec();
        let mask = mask[..seq_len].to_vec();

        let ids_tensor = Tensor::<i64>::from_array(([1usize, seq_len], ids))
            .map_err(|e| format!("input_ids tensor construction failed: {e}"))?;
        let mask_tensor = Tensor::<i64>::from_array(([1usize, seq_len], mask))
            .map_err(|e| format!("attention_mask tensor construction failed: {e}"))?;

        // Keep the RefMut alive long enough that `outputs` (which borrows from the
        // session's allocator) remains valid when we extract data from it.
        let mut session_guard = self.session.borrow_mut();
        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

        let (_shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("failed to extract sentence_embedding tensor: {e}"))?;

        let mut vec: Vec<f32> = data.to_vec();

        if vec.is_empty() {
            return Err("EmbeddingGemma returned zero-length embedding".to_string());
        }

        // L2-normalize so that cosine similarity reduces to a dot product.
        // This lets the HNSW hot loop skip the sqrt — half the FLOPS.
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }

        Ok(vec)
    }
}
