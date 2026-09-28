use tract_onnx::prelude::*;
use tract_nnef::prelude::*;
use std::fs;
use std::fs::File;
use anyhow::Context;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::collections::HashMap;
use tract_onnx::tract_hir::infer::Factoid;

fn convert_partition_to_nnef(
    onnx_path: &Path,
    nnef_out_path: &Path,
    weights_data: Option<&[u8]>,
    known_facts: &HashMap<String, InferenceFact>,
) -> TractResult<HashMap<String, InferenceFact>> {
    let nnef = tract_nnef::nnef().with_tract_core().with_onnx();

    println!("Full path: {}", onnx_path.display());
    let previous_onnx_path = onnx_path;
    let onnx_path = onnx_path
        .to_str()
        .unwrap()
        .rsplit_once('/')
        .map(|(_, file)| file)
        .unwrap_or_else(|| onnx_path.to_str().unwrap());
    let mut t = Instant::now();
    let mut model = tract_onnx::onnx().model_for_path(previous_onnx_path, weights_data)?;
    println!("[{}] onnx load = {:?}", onnx_path, t.elapsed());

    let batch_size = 1i64;
    let mut solver = tract_core::prelude::SymbolValues::default();
    for sym_name in ["batch_size", "batch"] {
        let sym = model.symbol_table.sym(sym_name);
        solver = solver.with(&sym, batch_size);
    }

    let is_gpt2 = onnx_path.contains("gpt2");
    for (i, outlet) in model.input_outlets()?.to_vec().into_iter().enumerate() {
        let name = model
            .outlet_label(outlet)
            .map(|s| s.to_string())
            .unwrap_or_else(|| model.node(outlet.node).name.clone());
        let fact = match known_facts.get(&name) {
            Some(f) => f.clone(),
            None => model.input_fact(i)?.clone(),
        };

        let fact = if is_gpt2 {
            match fact.shape.concretize() {
                Some(ds) => {
                    let dims: TVec<TDim> =
                        ds.iter().map(|d| d.eval(&solver)).collect();

                    InferenceFact::dt_shape(
                        fact.datum_type().unwrap(),
                        dims,
                    )
                }
                None => fact,
            }
        } else {
            fact
        };
        model.set_input_fact(outlet.node, fact)?;
    }

    // before optimizing: ONNX names, in order
    let label_of = |m: &InferenceModel, o: OutletId| -> String {
        m.outlet_label(o).map(|s| s.to_string()).unwrap_or_else(|| m.node(o.node).name.clone())
    };
    let in_names: Vec<String>  = model.input_outlets()?.iter().map(|o| label_of(&model, *o)).collect();
    let out_names: Vec<String> = model.output_outlets()?.iter().map(|o| label_of(&model, *o)).collect();

    t = Instant::now();
    let mut typed = model.into_typed()?;
    println!("[{}] onnx typed = {:?}", onnx_path, t.elapsed());

    if is_gpt2 {
        t = Instant::now();
        typed = typed.concretize_dims(&solver)?;
        println!("[{}] onnx concretize_dims = {:?}", onnx_path, t.elapsed());
    }

    t = Instant::now();
    let mut optimized = typed.into_optimized()?;
    println!("[{}] optimized = {:?}", onnx_path, t.elapsed());
    anyhow::ensure!(optimized.output_outlets()?.len() == out_names.len(),
                    "optimizer changed the number of outputs");

    // restore output names
    let old_outs: Vec<OutletId> = optimized.output_outlets()?.to_vec();
    let ins = optimized.input_outlets()?.to_vec();
    let mut new_outs: Vec<OutletId> = vec![];
    for (k, o) in old_outs.iter().enumerate() {
        let is_bare_input = ins.contains(o);
        let is_dup = old_outs[..k].contains(o);
        let outlet = if is_bare_input || is_dup {
            optimized.wire_node(&out_names[k], tract_core::ops::identity::Identity, &[*o])?[0]
        } else {
            *o
        };
        optimized.set_outlet_label(outlet, out_names[k].clone())?;
        new_outs.push(outlet);
    }
    optimized.set_output_outlets(&new_outs)?;
    for (k, o) in optimized.input_outlets()?.to_vec().iter().enumerate() {
        optimized.set_outlet_label(*o, in_names[k].clone())?;
    }

    let mut next_facts = known_facts.clone();
    for outlet in &optimized.outputs {
        let name = optimized
            .outlet_label(*outlet)
            .map(|s| s.to_string())
            .unwrap_or_else(|| optimized.node(outlet.node).name.clone());
        let fact = optimized.outlet_fact(*outlet)?;
        let dims: TVec<TDim> = fact.shape.to_tvec();
        next_facts.insert(name, InferenceFact::dt_shape(fact.datum_type, dims));
    }

    let file = fs::File::create(nnef_out_path)?;
    nnef.write_to_tar(&optimized, file)?;
    println!("saved {} -> {}", onnx_path, nnef_out_path.display());

    Ok(next_facts)
}

fn natural_cmp(a: &Path, b: &Path) -> std::cmp::Ordering {
    let a = a.to_string_lossy();
    let b = b.to_string_lossy();
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();

    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(&ca), Some(&cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let na: String = std::iter::from_fn(|| ai.by_ref().next_if(|c| c.is_ascii_digit())).collect();
                    let nb: String = std::iter::from_fn(|| bi.by_ref().next_if(|c| c.is_ascii_digit())).collect();
                    let na: u64 = na.parse().unwrap();
                    let nb: u64 = nb.parse().unwrap();
                    if na != nb {
                        return na.cmp(&nb);
                    }
                } else {
                    if ca != cb {
                        return ca.cmp(&cb);
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

fn open_weights_file(
    onnx_path: &Path,
) -> TractResult<Vec<u8>> {
    let full_path = onnx_path.join("model.onnx_data");

    let file = match File::open(&full_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Weights file not found at {:?}", full_path);
            return Ok(Vec::new());
        }
        Err(e) => return Err(e).context(format!("Opening {:?}", full_path)).map_err(Into::into),
    };
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap.to_vec())
}  

fn main() -> TractResult<()> {
    let model_names = ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"];
    let options = ["", "partitions_inter/", "partitions_intra/"];

    for name in model_names {
        for option in options {
            let mut model_dir = PathBuf::from(format!(
                "../../../github_repo/InferONNX/models/{name}"
            ));

            let decrypted = open_weights_file(&model_dir)?;
            let weights_data = if decrypted.is_empty() {
                None
            } else {
                Some(decrypted)
            };

            if !option.is_empty() {
                model_dir.push(option);
            }

            println!("Input directory: {}", model_dir.display());
            let mut partitions: Vec<PathBuf> = fs::read_dir(&model_dir)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "onnx"))
                .collect();
            partitions.sort_by(|a, b| natural_cmp(a, b));
            
            if partitions.is_empty() {
                println!("No ONNX files found, skipping.");
                continue;
            }
            for p in &partitions {
                println!("{}", p.display());
            }

            let dir = if name.contains("gpt2") {
                "gpt2"
            } else if name.contains("llama") {
                "llama"
            } else if name.contains("mistral") {
                "mistral"
            } else if name.contains("qwen") {
                "qwen"
            } else {
                return Err(anyhow::anyhow!("Unknown model: {name}!"));
            };
            let mut out_dir = PathBuf::from(dir);
            if !option.is_empty() {
                out_dir.push(option);
            }
            fs::create_dir_all(&out_dir)?;

            let is_partitioned = !option.is_empty();
            let mut facts = HashMap::new();
            for (i, onnx_path) in partitions.iter().enumerate() {
                let nnef_path = if is_partitioned {
                    out_dir.join(format!("{name}_split{i}.tar"))
                } else {
                    out_dir.join(format!("{name}.tar"))
                };

                println!("Converting {} -> {}", onnx_path.display(), nnef_path.display());
                facts = convert_partition_to_nnef(onnx_path, &nnef_path, weights_data.as_deref(), &facts)?;
            }
        }
    }

    Ok(())
}