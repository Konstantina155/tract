use ndarray::{s,Array2};
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use tokenizers::tokenizer::{Result, Tokenizer};
use tract_onnx::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::Regex;

use jemalloc_ctl::{epoch, stats};
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn albert(model_path: &str) -> Result<()> {
    let model_dir = PathBuf::from_str(model_path)?;
    let tokenizer = Tokenizer::from_file(Path::join(&model_dir, "tokenizer.json"))?;

    let text = "Paris is the [MASK] of France.";

    let tokenizer_output = tokenizer.encode(text, true)?;
    let input_ids = tokenizer_output.get_ids();
    let attention_mask = tokenizer_output.get_attention_mask();
    let token_type_ids = tokenizer_output.get_type_ids();
    let length = input_ids.len();
    let mask_pos =
        input_ids.iter().position(|&x| x == tokenizer.token_to_id("[MASK]").unwrap()).unwrap();

    let model = tract_onnx::onnx()
        .model_for_path(Path::join(&model_dir, "model.onnx"), None)?
        .into_optimized()?
        .into_runnable()?;

    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        input_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let attention_mask: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        attention_mask.iter().map(|&x| x as i64).collect(),
    )?
    .into();
    let token_type_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        token_type_ids.iter().map(|&x| x as i64).collect(),
    )?
    .into();

    let outputs =
        model.run(tvec!(input_ids.into(), attention_mask.into(), token_type_ids.into()))?;
    let logits = outputs[0].to_array_view::<f32>()?;
    let logits = logits.slice(s![0, mask_pos, ..]);
    let word_id = logits.iter().zip(0..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap()).unwrap().1;
    let word = tokenizer.id_to_token(word_id);
    println!("Albert: {word:?}");

    print_memory("Before drop");
    drop(model);
    drop(tokenizer);
    drop(tokenizer_output);
    drop(outputs);
    print_memory("After drop");

   Ok(())
}

fn gpt2(model_path: &str) -> Result<String> {
    let model_dir = PathBuf::from_str(model_path)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;

    let model = tract_onnx::onnx()
        .model_for_path(model_dir.join("model.onnx"), None)?
        .into_optimized()?
        .into_runnable()?;

    let prompt = "Hello, how are you today?";

    let tokenizer_output = tokenizer.encode(prompt, true)?;
    let mut current_ids: Vec<u32> = tokenizer_output.get_ids().to_vec();
    let mut current_attention_mask: Vec<u32> = tokenizer_output.get_attention_mask().to_vec();

    let max_tokens = 30;
    for _ in current_ids.len()..max_tokens {
        let input_ids_tensor: Tensor = Array2::from_shape_vec(
            (1, current_ids.len()),
            current_ids.iter().map(|&x| x as i64).collect(),
        )?.into();

        let attention_mask_tensor: Tensor = Array2::from_shape_vec(
            (1, current_attention_mask.len()),
            current_attention_mask.iter().map(|&x| x as i64).collect(),
        )?.into();

        let outputs = model.run(tvec!(input_ids_tensor.into(), attention_mask_tensor.into()))?;
        let logits = outputs[0].to_array_view::<f32>()?;
        let last_logits = logits.slice(s![0, -1, ..]);

        // Top-k sampling
        let k = 10;
        let mut scored: Vec<(usize, f32)> = last_logits
            .iter()
            .cloned()
            .enumerate()
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k = &scored[..k.min(scored.len())];
        let next_token_id = top_k
            .choose(&mut thread_rng())
            .map(|(idx, _)| *idx)
            .unwrap() as u32;

        // Stop if model outputs <|endoftext|> token (50256 in GPT-2)
        if next_token_id == 50256 {
            break;
        }

        current_ids.push(next_token_id);
        current_attention_mask.push(1);
    }

    let generated_text = tokenizer.decode(&current_ids, true)?;
    
    let sentence_regex = Regex::new(r"[^.!?]+[.!?]").unwrap();
    let sentences: Vec<&str> = sentence_regex
    .find_iter(&generated_text)
    .map(|m| m.as_str().trim())
    .take(3)
    .collect();

    let two_sentences = sentences.join(" ");
    println!("GPT2: {}", two_sentences);
    Ok(two_sentences)
}

fn latest_models(model_path: &str) -> Result<String> {
    let model_dir = PathBuf::from_str(model_path)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;

    let model_name = if let Some(pos) = model_path.rfind('/') {
        &model_path[pos + 1..]
    } else {
        model_path
    };
    let _model_name_with_ext = format!("{}.onnx", model_name);

    let model = tract_onnx::onnx()
        .model_for_path(model_dir.join("model.onnx"), None)?
        .into_optimized()?
        .into_runnable()?;

    let prompt = "Hello, how are you today?";

    let tokenizer_output = tokenizer.encode(prompt, true)?;
    let mut current_ids: Vec<u32> = tokenizer_output.get_ids().to_vec();
    let mut current_attention_mask: Vec<u32> = tokenizer_output.get_attention_mask().to_vec();
    let mut current_position_ids: Vec<u32> = (0..current_ids.len() as u32).collect();

    let max_tokens = 30;
    for _ in current_ids.len()..max_tokens {
        let input_ids_tensor: Tensor = Array2::from_shape_vec(
            (1, current_ids.len()),
            current_ids.iter().map(|&x| x as i64).collect(),
        )?.into();

        let attention_mask_tensor: Tensor = Array2::from_shape_vec(
            (1, current_attention_mask.len()),
            current_attention_mask.iter().map(|&x| x as i64).collect(),
        )?.into();

        let position_ids_tensor: Tensor = Array2::from_shape_vec(
            (1, current_position_ids.len()),
            current_position_ids.iter().map(|&x| x as i64).collect(),
        )?.into();

        let outputs = model.run(tvec!(input_ids_tensor.into(), attention_mask_tensor.into(), position_ids_tensor.into()))?;
        let logits = outputs[0].to_array_view::<f32>()?;
        let last_logits = logits.slice(s![0, -1, ..]);

        // Top-k sampling
        let k = 10;
        let mut scored: Vec<(usize, f32)> = last_logits
            .iter()
            .cloned()
            .enumerate()
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k = &scored[..k.min(scored.len())];
        let next_token_id = top_k
            .choose(&mut thread_rng())
            .map(|(idx, _)| *idx)
            .unwrap() as u32;

        // Stop if model outputs <|endoftext|> token (50256 in GPT-2)
        if next_token_id == 50256 {
            break;
        }

        current_ids.push(next_token_id);
        current_attention_mask.push(1);
        current_position_ids.push(current_position_ids.last().unwrap() + 1);

        print_memory("Before drop outputs");
        drop(outputs);
        print_memory("After drop outputs");
    }

    let generated_text = tokenizer.decode(&current_ids, true)?;
    
    let sentence_regex = Regex::new(r"[^.!?]+[.!?]").unwrap();
    let sentences: Vec<&str> = sentence_regex
    .find_iter(&generated_text)
    .map(|m| m.as_str().trim())
    .take(3)
    .collect();

    let two_sentences = sentences.join(" ");
    let name: &str = Path::new(model_path)
        .file_name()
        .and_then(|s| s.to_str())
        .expect("Invalid path or non-UTF8 filename");
    println!("{}: {}", name, two_sentences);

    print_memory("Before drop");
    drop(model);
    drop(tokenizer);
    drop(tokenizer_output);
    print_memory("After drop");

    Ok(two_sentences)
}

fn print_memory(label: &str) {
    epoch::advance().unwrap();
    let allocated = stats::allocated::read().unwrap();
    println!("[{label}] Memory used: {} bytes", allocated);
}

fn main() -> Result<()> {
    // print_memory("Start1 cerebras-gpt");
    // latest_models("./cerebras-gpt")?;
    // print_memory("After1 cerebras-gpt");

    // print_memory("Start2 cerebras-gpt");
    // latest_models("./cerebras-gpt")?;
    // print_memory("After2 cerebras-gpt");

    print_memory("Start1 albert");
    albert("./albert")?;
    print_memory("After1 albert");

    // albert("./albert")?;
    // latest_models("./cerebras-gpt")?;
    // latest_models("../../../github_repo/InferONNX/models/qwen2.5-0.5B")?;
    // latest_models("../../../github_repo/InferONNX/models/deepseek-coder-1.3b-base")?;
    // latest_models("../../../github_repo/InferONNX/models/llama3.2-1B")?;
    Ok(())
}
