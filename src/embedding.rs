use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let device = Device::Cpu;
        let repo_id = "intfloat/multilingual-e5-small";

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.as_ref().to_path_buf())
            .build()?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        eprintln!("Loading model {}...", repo_id);

        let config_path = repo.get("config.json").context("downloading config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("downloading tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("downloading model.safetensors")?;

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!("{}", e))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
        };
        let model = BertModel::load(vb, &config)?;

        eprintln!("Model loaded.");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        let token_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let attention_mask_t = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let output = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask_t))?;

        // Mean pooling over token dimension, masked by attention
        let mask = attention_mask_t
            .unsqueeze(2)?
            .to_dtype(candle_core::DType::F32)?;
        let masked = output.broadcast_mul(&mask)?;
        let sum = masked.sum(1)?;
        let count = mask.sum(1)?;
        let mean_pooled = sum.broadcast_div(&count)?;

        // L2 normalize
        let norm = mean_pooled
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?;
        let normalized = mean_pooled.broadcast_div(&norm)?;

        let embedding: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        Ok(embedding)
    }

    /// Embed a context label (document/passage side).
    pub fn embed_context(&self, label: &str) -> Result<Vec<f32>> {
        self.embed(&format!("passage: {}", label))
    }

    /// Embed a query string (query side).
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(&format!("query: {}", query))
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Serialize f32 vector to bytes for SQLite BLOB storage.
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Deserialize bytes from SQLite BLOB to f32 vector.
pub fn bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
