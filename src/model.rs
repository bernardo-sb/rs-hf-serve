use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer, TokenizerImpl,
};

pub struct Bert {
    model: BertModel,
    tokenizer: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    device: Device,
}

impl Bert {
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        use_torch: bool,
        approximate_gelu: bool,
    ) -> Result<Bert> {
        let device = Device::Cpu;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (model_id.to_owned(), revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);

        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if use_torch {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Set padding and truncation to None
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)
            .unwrap();

        let tokenizer = &*tokenizer; // Get as immutable reference

        let vb = if use_torch {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }

        let model = BertModel::load(vb, &config)?;
        let tokenizer = tokenizer.clone();

        Ok(Bert {
            model: model,
            tokenizer: tokenizer,
            device: device,
        })
    }

    pub fn predict(&self, prompt: String) -> Result<Tensor> {
        let start = std::time::Instant::now();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        println!("Loaded and encoded {:?}", start.elapsed());

        let ys = self.model.forward(&token_ids, &token_type_ids, None)?;
        println!("Forward pass took {:?}", start.elapsed());

        Ok(ys)
    }
}
