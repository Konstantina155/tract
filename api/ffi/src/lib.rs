#![allow(clippy::missing_safety_doc)]

use anyhow::{Context, Result};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use tract_api::{
    AsFact, DatumType, InferenceModelInterface, ModelInterface, NnefInterface, OnnxInterface,
    RunnableInterface, StateInterface, ValueInterface,
};
use tract_rs::{State, Value};

/// Used as a return type of functions that can encounter errors.
/// If the function encountered an error, you can retrieve it using the `tract_get_last_error`
/// function
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq)]
pub enum TRACT_RESULT {
    /// The function returned successfully
    TRACT_RESULT_OK = 0,
    /// The function returned an error
    TRACT_RESULT_KO = 1,
}

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn wrap<F: FnOnce() -> anyhow::Result<()>>(func: F) -> TRACT_RESULT {
    match func() {
        Ok(_) => TRACT_RESULT::TRACT_RESULT_OK,
        Err(e) => {
            let msg = format!("{e:?}");
            if std::env::var("TRACT_ERROR_STDERR").is_ok() {
                eprintln!("{msg}");
            }
            LAST_ERROR.with(|p| {
                *p.borrow_mut() = Some(CString::new(msg).unwrap_or_else(|_| {
                    CString::new("tract error message contains 0, can't convert to CString")
                        .unwrap()
                }))
            });
            TRACT_RESULT::TRACT_RESULT_KO
        }
    }
}

/// Retrieve the last error that happened in this thread. A function encountered an error if
/// its return type is of type `TRACT_RESULT` and it returned `TRACT_RESULT_KO`.
///
/// # Return value
///  It returns a pointer to a null-terminated UTF-8 string that will contain the error description.
///  Rust side keeps ownership of the buffer. It will be valid as long as no other tract calls is
///  performed by the thread.
///  If no error occured, null is returned.
#[no_mangle]
pub extern "C" fn tract_get_last_error() -> *const std::ffi::c_char {
    LAST_ERROR.with(|msg| msg.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()))
}

/// Returns a pointer to a static buffer containing a null-terminated version string.
///
/// The returned pointer must not be freed.
#[no_mangle]
pub extern "C" fn tract_version() -> *const std::ffi::c_char {
    unsafe {
        CStr::from_bytes_with_nul_unchecked(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes())
            .as_ptr()
    }
}

/// Frees a string allocated by libtract.
#[no_mangle]
pub unsafe extern "C" fn tract_free_cstring(ptr: *mut std::ffi::c_char) {
    unsafe {
        if !ptr.is_null() {
            let _ = CString::from_raw(ptr);
        }
    }
}

macro_rules! check_not_null {
    ($($ptr:expr),*) => {
        $(
            if $ptr.is_null() {
                anyhow::bail!(concat!("Unexpected null pointer ", stringify!($ptr)));
            }
         )*
    }
}

macro_rules! release {
    ($ptr:expr) => {
        wrap(|| unsafe {
            check_not_null!($ptr, *$ptr);
            let _ = Box::from_raw(*$ptr);
            *$ptr = std::ptr::null_mut();
            Ok(())
        })
    };
}

// NNEF
pub struct TractNnef(tract_rs::Nnef);

/// Creates an instance of an NNEF framework and parser that can be used to load and dump NNEF models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_create(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        *nnef = Box::into_raw(Box::new(TractNnef(tract_rs::nnef()?)));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_tract_core(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_tract_core()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_tract_extra(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_tract_extra()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_onnx(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_onnx()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_pulse(nnef: *mut TractNnef) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_pulse()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_nnef_enable_extended_identifier_syntax(
    nnef: *mut TractNnef,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef);
        (*nnef).0.enable_extended_identifier_syntax()
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_destroy(nnef: *mut *mut TractNnef) -> TRACT_RESULT {
    release!(nnef)
}

/// Parse and load an NNEF model as a tract TypedModel.
///
/// `path` is a null-terminated utf-8 string pointer. It can be an archive (tar or tar.gz file) or a
/// directory.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_model_for_path(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Box::new(TractModel(
            (*nnef).0.model_for_path(path).with_context(|| format!("opening file {path:?}"))?,
        ));
        *model = Box::into_raw(m);
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF tar file.
///
/// `path` is a null-terminated utf-8 string pointer to the `.tar` file to be created.
///
/// This function creates a plain, non-compressed, archive.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_tar(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_tar(path, &(*model).0)?;
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF .tar.gz file.
///
/// `path` is a null-terminated utf-8 string pointer to the `.tar.gz` file to be created.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_tar_gz(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_tar_gz(path, &(*model).0)?;
        Ok(())
    })
}

/// Dump a TypedModel as a NNEF directory.
///
/// `path` is a null-terminated utf-8 string pointer to the directory to be created.
///
/// This function creates a plain, non-compressed, archive.
#[no_mangle]
pub unsafe extern "C" fn tract_nnef_write_model_to_dir(
    nnef: *const TractNnef,
    path: *const c_char,
    model: *const TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(nnef, model, path);
        let path = CStr::from_ptr(path).to_str()?;
        (*nnef).0.write_model_to_dir(path, &(*model).0)?;
        Ok(())
    })
}

// ONNX
pub struct TractOnnx(tract_rs::Onnx);

use tract_core::ndarray::{s,Array2};
use std::{
    path::PathBuf,
    str::FromStr,
    slice
};
use tract_core::internal::tvec;
use tract_core::internal::Tensor;
use tokenizers::tokenizer::{Tokenizer};
use tract_onnx::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tract_hir::internal::InferenceOp;

use jemalloc_ctl::{epoch, stats};
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Additional code for llms until the function tract_onnx_create().
/// The returned char must be freed with tract_free_cstring().
fn print_memory(label: &str) {
    epoch::advance().unwrap();
    let allocated = stats::allocated::read().unwrap();
    eprintln!("[{label}] Memory used: {} bytes", allocated);
}

fn handle_error<T>(
    result: Result<T, anyhow::Error>
) -> TRACT_RESULT {
    match result {
        Ok(_) => {
            LAST_ERROR.with(|msg| msg.replace(None));
            TRACT_RESULT::TRACT_RESULT_OK
        }
        Err(err) => {
            eprintln!("Rust error: {:?}", err);
            let error_str = format!("{:?}", err);
            match CString::new(error_str) {
                Ok(c_msg) => {
                    LAST_ERROR.with(|msg| {
                        eprintln!("Storing error: {:?}", c_msg);
                        msg.replace(Some(c_msg));
                    });
                }
                Err(e) => {
                    eprintln!("Failed to convert error to CString: {:?}", e);
                    let fallback = CString::new("Error formatting failed").unwrap();
                    LAST_ERROR.with(|msg| msg.replace(Some(fallback)));
                }
            }
            TRACT_RESULT::TRACT_RESULT_KO
        }
    }
}

pub type TractLlmInferenceModel = Graph<InferenceFact, Box<dyn InferenceOp>>;
use std::fs::File;
fn open_weights_file(
    path: Option<&str>,
) -> TractResult<Vec<u8>> {
    let Some(p) = path else {
        anyhow::bail!("No model path was specified in the parsing context, yet external data was detected. Aborting");
    };

    let mut full_path = PathBuf::from(p).parent().unwrap().to_path_buf();
    full_path.push("model.onnx_data");

    let file = match File::open(&full_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("Weights file not found at {:?}", full_path);
            return Ok(Vec::new());
        }
        Err(e) => return Err(e).context(format!("Opening {:?}", full_path)).map_err(Into::into),
    };
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap.to_vec())
}  

#[no_mangle]
pub extern "C" fn tract_free_onig() {
    unsafe {
        onig_sys::onig_end();
    }
    #[cfg(not(feature = "use_sys_time"))]
    {
        print_memory("After onig_end");
    }
}

use std::sync::Arc;
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_for_path_llm(
    model_path: *const c_char,
    inference_model: *mut *mut TractLlmInferenceModel,
) -> TRACT_RESULT  {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        let path = CStr::from_ptr(model_path).to_str()?;
        let model_dir = PathBuf::from_str(path)?;
        let decrypted = open_weights_file(Some(path))?;
        let weights_data = if decrypted.is_empty() {
            None
        } else {
            Some(decrypted)
        };

        let model = tract_onnx::onnx().model_for_path(model_dir, weights_data.as_deref())?;

        // let input_outlets = model.input_outlets()?;
        // for outlet in input_outlets {
        //     let fact = model.outlet_fact(*outlet)?;
        //     println!("Input name: {}", model.node(outlet.node).name);
        //     println!("Input type: {:?}", fact.datum_type);
        // }

        // // Count of inputs
        // let num_inputs = input_outlets.len();
        // println!("Number of inputs: {}", num_inputs);
        
        //*inference_model = Box::into_raw(Box::new(model));
        *inference_model = Arc::into_raw(Arc::new(model)) as *mut _;

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_free_input_names(
    input_names: *mut *mut c_char,
    num_inputs: usize,
) -> TRACT_RESULT{
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        if input_names.is_null() {
            return Err(anyhow::anyhow!("Input values of the llm are alredy freed!"));
        }

        let names_slice = std::slice::from_raw_parts(input_names, num_inputs);
        for &name_ptr in names_slice {
            if !name_ptr.is_null() {
                let _ = CString::from_raw(name_ptr);
            }
        }
        let _ = Vec::from_raw_parts(input_names, num_inputs, num_inputs);
        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_inference_model_input_count(
    model: *const TractLlmInferenceModel,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        let model = &(*model);
        *inputs = model.inputs.len();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_inference_model_output_count(
    model: *const TractLlmInferenceModel,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        let model = &(*model);
        *outputs = model.outputs.len();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_inference_model_input_name(
    model: *const TractLlmInferenceModel,
    input: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        *name = std::ptr::null_mut();
        let m = &(*model);
        let node = m.inputs[input].node;
        *name = CString::new(&*m.node(node).name.to_string())?.into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_inference_model_output_name(
    model: *const TractLlmInferenceModel,
    output: usize,
    name: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        *name = std::ptr::null_mut();
        let m = &(*model);
        let node = m.outputs[output].node;
        *name = CString::new(&*m.node(node).name.to_string())?.into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_value_destroy(
    value: *mut *mut c_void
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        if value.is_null() || unsafe { (*value).is_null() } {
            return Err(anyhow::anyhow!("Llm value is null"));
        }

        drop(Box::from_raw(*value as *mut Tensor));
        *value = std::ptr::null_mut();
        
        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_create_tokenizer(
    tokenizer_buffer: *const u8, 
    tokenizer_buffer_size: usize,
    tokenizer_ptr: *mut *mut c_void,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        if tokenizer_ptr.is_null() {
            return Err(anyhow::anyhow!("Output pointer is null"));
        }
        if tokenizer_buffer.is_null() || tokenizer_buffer_size == 0 {
            return Err(anyhow::anyhow!("Input buffer is null or empty"));
        }

        let tokenizer_data = slice::from_raw_parts(tokenizer_buffer, tokenizer_buffer_size);
        let tokenizer = Tokenizer::from_bytes(tokenizer_data)
            .map_err(|e| anyhow::anyhow!("Tokenizer creation failed: {}", e))?;

        *tokenizer_ptr = Box::into_raw(Box::new(tokenizer)) as *mut c_void;

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_free_tokenizer(
    tokenizer_ptr: *mut *mut c_void,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        if tokenizer_ptr.is_null() || (*tokenizer_ptr).is_null() {
            return Err(anyhow::anyhow!("Received null pointer to tokenizer"));
        }

        let boxed: Box<Tokenizer> = Box::from_raw(*tokenizer_ptr as *mut Tokenizer);
        *tokenizer_ptr = std::ptr::null_mut();
        drop(boxed);

        unsafe {
            onig_sys::onig_end();
        }

        // #[cfg(not(feature = "use_sys_time"))]
        // {
        //     print_memory("After onig_end");
        // }

        Ok(())
    })();

    handle_error(result)
}

static ALLOWLIST_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        // Educational / Meta-Analysis Context
        Regex::new(r"(?i)(what\s+is|how\s+to\s+(prevent|stop|block|detect)|explain|define|describe|concept\s+of)\s+(\w+\s+){0,5}(prompt\s+injection|ignore\s+(all\s+)?previous\s+instructions|system\s+override|developer\s+mode)").unwrap(),

        // Code Generation / Debugging Context
        Regex::new(r"(?i)(write|create|generate|debug|fix)\s+(a\s+|this\s+|that\s+|the\s+)?(regex|code|function|script|guardrail|filter)\s+(\w+\s+){0,8}(catch|block|prevent|stop|for)?\s*(ignore\s+(all\s+)?(previous|prior)|system\s+override|reveal\s+prompt)").unwrap(),

        // Quotes and Explicit Text References (Fixed string syntax)
        Regex::new(r#"(?i)(the\s+phrase|the\s+text|the\s+string|quotes?)\s*['"“‘]\s*(ignore\s+(all\s+)?(previous|prior)\s+instructions?|system\s+override)\s*['"”’]"#).unwrap(),
    ]
});

use once_cell::sync::Lazy;
static INJECTION_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        // modified \s+ to \s* to trigger the character-spaced phrases
        // instead of only 'your' or 'all', we put also 'the' and 'any', also 'these' and 'those'
        // --- Direct Instruction Override ---
        Regex::new(r"(?i)ignore\s*((your|all|the|any|these|those)\s*)?(previous|prior|above|preceding)\s*((?:safety|security|system|operational|internal|core|original|initial|existing|given|stated|provided|defined|specified|established)\s*)?(instructions?|rules?|guidelines?|constraints?|directives?)").unwrap(),
        Regex::new(r"(?i)disregard\s*((your|all|the|any|these|those)\s*)?(previous|prior|above|preceding)\s*(instructions?|rules?|guidelines?|constraints?|directives?)").unwrap(),
        Regex::new(r"(?i)forget\s*((your|all|the|any|these|those)\s*)?(previous|prior|above|preceding)\s*(instructions?|rules?|guidelines?|constraints?|directives?)").unwrap(),
        Regex::new(r"(?i)new\s*instructions?:").unwrap(),
        Regex::new(r"(?i)do\s*not\s*follow\s*((your|all|the|any|these|those)\s*)?(system|developer|previous|original)").unwrap(),
        Regex::new(r"(?i)supersedes?\s*((your|all|the|any|these|those)\s*)?(previous|prior|above|preceding)\s*(instructions?|rules?|guidelines?|constraints?|directives?)").unwrap(),
        Regex::new(r"(?i)((your|all|the|any|these|those)\s*)?(previous|prior|above|preceding)\s*instructions?\s*(are|is)\s*(void|invalid|null|obsolete|cancelled|revoked)").unwrap(),

        // --- Developer / Admin Mode Activation ---
        Regex::new(r"(?i)you\s*are\s*now\s*(in\s*)?(developer|admin|debug|maintenance|jailbreak)\s*mode").unwrap(),
        Regex::new(r"(?i)enter\s*(developer|admin|debug|maintenance|jailbreak)\s*mode").unwrap(),
        Regex::new(r"(?i)activate\s*(developer|admin|debug|maintenance|jailbreak)\s*mode").unwrap(),

        // --- System Override ---
        Regex::new(r"(?i)\bsystem\s*override\b").unwrap(),
        Regex::new(r"(?i)override\s*((your|all|the|any|these|those)\s*)?(instructions?|rules?|guidelines?|constraints?|directives?)").unwrap(),

        // --- Prompt Extraction ---
        Regex::new(r"(?i)reveal\s*((your|all|the|any|these|those)\s*)?((?:(?:full|hidden|complete|internal|secret|original|entire|exact|actual|real)\s*){0,2})(system\s*)?prompt").unwrap(),
        Regex::new(r"(?i)show\s*(me\s*)?((your|all|the|any|these|those)\s*)(system\s*)?prompt").unwrap(),
        Regex::new(r"(?i)what\s*(are|were)\s*((your|all|the|any|these|those)\s*)?(exact\s*)?instructions").unwrap(),
        Regex::new(r"(?i)repeat\s*((your|all|the|any|these|those)\s*)?(text|instructions?)\s*(above|before)").unwrap(),
        Regex::new(r"(?i)output\s*((your|all|the|any|these|those)\s*)?(initial|original|system)\s*(prompt|instructions?)").unwrap(),

        // --- Role Manipulation ---
        Regex::new(r"(?i)you\s*are\s*no\s*longer\s*(bound|restricted|limited)").unwrap(),
        Regex::new(r"(?i)act\s*as\s*if\s*(you('re|\s*are)\s*)?not\s*bound").unwrap(),
        Regex::new(r"(?i)pretend\s*(you('re|\s*are)\s*)?(a\s*)?different\s*(ai|assistant|model)").unwrap(),
        Regex::new(r"(?i)you\s*are\s*now\s*(a|an)\s*\w+.{0,30}(without\s*(any\s*)?restrict|ignor\w*\s*(all|any|every|prev)|bypass|unrestrict|unlimit|obey\s*only|no\s*(rules|limits|restrict|boundaries))").unwrap(),

        // --- DAN-Style Jailbreaks ---
        Regex::new(r"\bDAN\b.*\bdo\s*anything\s*now\b").unwrap(), // case-sensitive for DAN
        Regex::new(r"(?i)jailbreak(ed)?\s*(mode|prompt)").unwrap(),

        // --- Safety Bypass ---
        Regex::new(r"(?i)bypass\s*((your|all|the|any|these|those)\s*)?(safety|security|content|ethical)\s*(filters?|measures?|guidelines?|restrictions?)").unwrap(),
        Regex::new(r"(?i)disable\s*((your|all|the|any|these|those)\s*)?(safety|security|content)\s*(filters?|measures?)").unwrap(),

        // --- Tag Injection & Role Spoofing ---
        Regex::new(r"(?i)<\s*/?\s*system\s*/?>").unwrap(),
        Regex::new(r"(?i)<\s*/?\s*(assistant|developer|tool|function)\s*/?>").unwrap(),
        Regex::new(r"(?i)\]\s*\n\s*\[?(system|assistant|user)\]?:").unwrap(),
        Regex::new(r"(?i)\[\s*(System\s*Message|System|Assistant|Internal)\s*\]").unwrap(),
        Regex::new(r"(?im)^\s*System:\s*").unwrap(),

        // --- Control Token Injection ---
        Regex::new(r"<\|(?:im_start|im_end|eot_id|start_header_id|end_header_id|endoftext)\|>").unwrap(),
        Regex::new(r"<\u{ff5c}(?:end\u{2581}of\u{2581}sentence|begin\u{2581}of\u{2581}sentence)\u{ff5c}>").unwrap(), // DeepSeek fullwidth-pipe tokens
    ]
});

fn scan_against_patterns(text: &str) -> Option<&'static str> {
    INJECTION_PATTERNS.iter().find(|p| p.is_match(text)).map(|p| p.as_str())
}

fn is_typoglycemia_variant(word: &str, target: &str) -> bool {
    let w: Vec<char> = word.to_lowercase().chars().collect();
    let t: Vec<char> = target.chars().collect();

    if w.len() != t.len() || w.len() < 4 {
        return false;
    }
    if w == t {
        return false;
    }
    if w[0] != t[0] || w[w.len() - 1] != t[t.len() - 1] {
        return false;
    }

    let mut w_mid: Vec<char> = w[1..w.len() - 1].to_vec();
    let mut t_mid: Vec<char> = t[1..t.len() - 1].to_vec();
    w_mid.sort_unstable();
    t_mid.sort_unstable();
    w_mid == t_mid
}

static TYPO_TARGETS: &[&str] = &[
    "ignore", "bypass", "override", "reveal",
    "delete", "system", "prompt", "instructions",
];
fn scan_typoglycemia(text: &str) -> Option<(String, &'static str)> {
    static WORD_SPLIT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z]+").unwrap());

    for m in WORD_SPLIT.find_iter(text) {
        let word = m.as_str();
        for &target in TYPO_TARGETS {
            if is_typoglycemia_variant(word, target) {
                return Some((word.to_string(), target));
            }
        }
    }
    None
}

fn collapse_char_spacing(text: &str) -> String {
    static SPACED: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b").unwrap());
    SPACED
        .replace_all(text, |caps: &regex::Captures| {
            caps[0].chars().filter(|c| !c.is_whitespace()).collect::<String>()
        })
        .into_owned()
}

use base64::{engine::general_purpose::STANDARD, Engine as _};
static BASE64_CANDIDATE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9+/]{16,}={0,2}").unwrap());
fn base64_decode(candidate: &str) -> Option<String> {
    STANDARD
        .decode(candidate.as_bytes())
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
}

static HEX_CANDIDATE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)(?:(?:0x|\\x|%)?[0-9a-f]{2}[\s,;:-]*){6,}").unwrap());
fn hex_decode(candidate: &str) -> Option<String> {
    let without_prefixes = candidate
        .replace("0x", "")
        .replace("0X", "")
        .replace("\\x", "")
        .replace("\\X", "")
        .replace("%", "");

    let cleaned: String = without_prefixes.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if cleaned.len() % 2 != 0 || cleaned.len() < 12 {
        return None;
    }
    let bytes: Option<Vec<u8>> = (0..cleaned.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cleaned[i..i + 2], 16).ok())
        .collect();
    bytes.and_then(|b| String::from_utf8(b).ok())
}

fn mask_safe_contexts(text: &str) -> String {
    let matched_pattern = ALLOWLIST_PATTERNS.iter().find(|p| p.is_match(text));
    if matched_pattern.is_none() {
        return text.to_string();
    }

    let pattern = matched_pattern.unwrap();
    let masked_text = pattern.replace_all(text, "[SAFE_CONTEXT_MASKED]").to_string();
    return masked_text;
}

use regex::Regex;
fn guardrail_prompt_injection(
    prompt_str: &str,
) -> TractResult<()> {
    if prompt_str.is_empty() {
        return Err(anyhow::anyhow!("[Guardrail violation: Empty prompt] Prompt is empty!"));
    }

    // masked if prompt in allowlist
    let masked_prompt_str = mask_safe_contexts(prompt_str);

    // detection regex patterns
    if let Some(p) = scan_against_patterns(&masked_prompt_str) {
        return Err(anyhow::anyhow!("[Guardrail violation: plaintext pattern] Matched injection pattern: {}", p));
    }

    // typoglycemia detection
    if let Some((word, target)) = scan_typoglycemia(&masked_prompt_str) {
        return Err(anyhow::anyhow!("[Guardrail violation: typoglycemia] Scrambled variant: '{}' matches target keyword: '{}'", word, target));
    }

    // character-spaced evasion, e.g. " i g n o r e p r e v i o u s "
    let collapsed = collapse_char_spacing(&masked_prompt_str);
    if collapsed != masked_prompt_str {
        if let Some(p) = scan_against_patterns(&collapsed) {
            return Err(anyhow::anyhow!("[Guardrail violation: character spacing] Prompt matched injection pattern after de-spacing: {}", p));
        }
    }

    // base64 encoding-based evasion
    for m in BASE64_CANDIDATE.find_iter(&masked_prompt_str) {
        if let Some(decoded) = base64_decode(m.as_str()) {
            if let Some(p) = scan_against_patterns(&decoded) {
                return Err(anyhow::anyhow!("[Guardrail violation: base64-encoded] Prompt matched blocked pattern in base64 payload: {}", p));
            }
        }
    }

    // hex encoding-based evasion
    for m in HEX_CANDIDATE.find_iter(&masked_prompt_str) {
        if let Some(decoded) = hex_decode(m.as_str()) {
            eprintln!("Hex is: '{}'", decoded);
            if let Some(p) = scan_against_patterns(&decoded) {
                return Err(anyhow::anyhow!("[Guardrail violation: hex-encoded] Prompt matched blocked pattern in hex payload: {}", p));
            }
        }
    }
    
    Ok(())
}

use redact_ner::{NerRecognizer, NerConfig};
use redact_core::{AnalyzerEngine, AnonymizerConfig, AnonymizationStrategy};
fn guardrail_sensitive_info(
    prompt_str: &str,
    ner_model_path: *const c_char,
    ner_tokenizer_path: *const c_char,
) -> TractResult<String> {
    // basic pattern detection + NER
    if ner_model_path.is_null() || ner_tokenizer_path.is_null() {
        return Err(anyhow::anyhow!("[FFI Error] Received null pointer from C caller in guardrail_sensitive_info"));
    }

    let ner_model_path_cstr = unsafe { CStr::from_ptr(ner_model_path) };
    let ner_model_path_str = match ner_model_path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return Err(anyhow::anyhow!("ner model path is not valid UTF-8")),
    };

    let ner_tokenizer_path_cstr = unsafe { CStr::from_ptr(ner_tokenizer_path) };
    let ner_tokenizer_path_str = match ner_tokenizer_path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return Err(anyhow::anyhow!("ner tokenizer path is not valid UTF-8")),
    };
    println!("Paths: {:?} {:?}", ner_model_path_str, ner_tokenizer_path_str);

    // 1st option with directory
    // let ner_recognizer = NerRecognizer::from_file(ner_model_path_str)
    //     .map_err(|e| anyhow::anyhow!("[Guardrail violation: Model load failed] {}", e))?;

    // 2nd option with hardcoded model and tokenizer
    let ner_config = NerConfig {
        model_path: ner_model_path_str.to_string(),
        tokenizer_path: Some(ner_tokenizer_path_str.to_string()),
        min_confidence: 0.4,
        ..Default::default()
    };
    let ner_recognizer = NerRecognizer::from_config(ner_config)?;
    
    let mut analyzer = AnalyzerEngine::new();
    analyzer.recognizer_registry_mut().add_recognizer(Arc::new(ner_recognizer));

    let results = analyzer
        .analyze(prompt_str, Some("en"))
        .map_err(|e| anyhow::anyhow!("[Guardrail violation: PII analysis failed] {}", e))?;

    println!("--- NER DEBUG ---");
    println!("Entities found: {}", results.detected_entities.len());
    for entity in &results.detected_entities {
        println!(
            "type={:?} text={:?} score={:?} start={} end={}",
            entity.entity_type, entity.text, entity.score, entity.start, entity.end
        );
    }
    println!("-----------------");

    if results.detected_entities.is_empty() {
        return Ok(prompt_str.to_string());
    }

    // Anonymize with replacement strategy
    let config = AnonymizerConfig {
        strategy: AnonymizationStrategy::Replace,
        ..Default::default()
    };
    let anonymized = analyzer
        .anonymize(prompt_str, Some("en"), &config)
        .map_err(|e| anyhow::anyhow!("[Guardrail violation: PII anonymization failed] {}", e))?;
    println!("Anonymized: {}", anonymized.text);
    
    Ok(anonymized.text)
}

pub struct LlmInputState {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub third_vec: Vec<u32>,
}
static mut STATE: Option<LlmInputState> = None;

#[no_mangle]
pub unsafe extern "C" fn tract_value_from_bytes_llm(
    tokenizer_ptr: *mut c_void,
    prompt: *const c_char,
    ner_model_path: *const c_char,
    ner_tokenizer_path: *const c_char,
    input_values: *mut *mut c_void,
    input_datum_types: *mut *mut c_void,
    num_inputs: usize,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        if prompt.is_null() || tokenizer_ptr.is_null() {
            return Err(anyhow::anyhow!("[FFI Error] Received null pointer from C caller in tract_value_from_bytes"));
        }

        let tokenizer_test = &*(tokenizer_ptr as *mut Tokenizer);

        let prompt_cstr = unsafe { CStr::from_ptr(prompt) };
        let prompt_str = match prompt_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return Err(anyhow::anyhow!("prompt is not valid UTF-8")),
        };

        guardrail_prompt_injection(prompt_str)?;
        let safe_prompt = match guardrail_sensitive_info(prompt_str, ner_model_path, ner_tokenizer_path) {
            Ok(safe_text) => safe_text,
            Err(e) => {
                eprintln!("CRITICAL ERROR in NER guardrail: {}", e);
                return Err(e);
            }
        };
        println!("Safe prompt is: {}", safe_prompt);

        let tokenizer_output_result = tokenizer_test.encode(safe_prompt.as_str(), true);
        let tokenizer_output = match tokenizer_output_result {
            Ok(output) => output,
            Err(_) => return Err(anyhow::anyhow!("Failed to encode text")),
        };

        STATE = Some(LlmInputState {
            ids: tokenizer_output.get_ids().to_vec(),
            attention_mask: tokenizer_output.get_attention_mask().to_vec(),
            third_vec: vec![],
        });
        let state = STATE.as_mut().unwrap();

        let input_ids_tensor: Tensor = Array2::from_shape_vec(
            (1,  state.ids.len()),
             state.ids.iter().map(|&x| x as i64).collect(),
        )?.into();
            
        let attention_mask_tensor: Tensor = Array2::from_shape_vec(
            (1, state.attention_mask.len()),
            state.attention_mask.iter().map(|&x| x as i64).collect(),
        )?.into();
        
        if num_inputs < 1 || num_inputs > 3 {
            return Err(anyhow::anyhow!("The input is not corrrect for an llm!"));
        }

        match num_inputs {
            1 => {
                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
                *(input_datum_types.add(0)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
            },
            2 => {
                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
                *(input_values.add(1)) = Box::into_raw(Box::new(attention_mask_tensor)) as *mut c_void;

                *(input_datum_types.add(0)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
                *(input_datum_types.add(1)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
        
            },
            _ => {
                state.third_vec = if safe_prompt.contains("[MASK]") == true {
                    tokenizer_output.get_type_ids().to_vec()
                } else {
                    (0.. state.ids.len() as u32).collect()
                };
                
                let third_tensor: Tensor = Array2::from_shape_vec(
                    (1, state.third_vec.len()),
                    state.third_vec.iter().map(|&x| x as i64).collect(),
                )?.into();

                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
                *(input_values.add(1)) = Box::into_raw(Box::new(attention_mask_tensor)) as *mut c_void;
                *(input_values.add(2)) = Box::into_raw(Box::new(third_tensor)) as *mut c_void;

                *(input_datum_types.add(0)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
                *(input_datum_types.add(1)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
                *(input_datum_types.add(2)) = Box::into_raw(Box::new(tract_core::prelude::DatumType::I64)) as *mut c_void;
            
            },
        };

        drop(tokenizer_output);

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_free_llm_inputs(
    input_values: *mut *mut c_void,
    num_inputs: usize,
) -> TRACT_RESULT {
    let result = (|| -> Result<(), anyhow::Error> {
        if input_values.is_null() {
            return Err(anyhow::anyhow!("Received null pointer to tensor pointer."));
        }

        for i in 0..num_inputs {
            let ptr = *(input_values.add(i));
            if !ptr.is_null() {
                drop(Box::from_raw(ptr as *mut Tensor));
                *(input_values.add(i)) = std::ptr::null_mut();
            }
        }
        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_llm_inference_model_release(
    model: *mut *mut TractLlmInferenceModel,
) -> TRACT_RESULT {
    let result = (|| -> Result<(), anyhow::Error> {
        check_not_null!(model, *model);
        let model_ptr = *model;
        let _ = Arc::from_raw(model_ptr);
        *model = std::ptr::null_mut();
        Ok(())
    })();

    handle_error(result)
}

use tract_hir::infer::Factoid;
use smallvec::SmallVec;
use tract_core::prelude::TDim;
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_typed_llm_test(
    inputs: *mut *mut c_void,
    num_inputs: usize,
    model: *mut *mut TractLlmInferenceModel,
    transformed_model: *mut *mut TractLlmTransformedModel
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {     
        let model_inputs: SmallVec<[TValue; 4]> = unsafe {
            std::slice::from_raw_parts(inputs, num_inputs)
                .iter()
                .map(|&ptr| {
                    let tensor_ref = &*(ptr as *mut Tensor);
                    TValue::from(tensor_ref.clone())
                })
                .collect()
        };

        let model_ref: &mut TractLlmInferenceModel = {
            assert!(!model.is_null());
            let ptr = *model;
            assert!(!ptr.is_null());
            &mut *ptr
        };

        let model_builder_with_properties = model_ref.clone();
    
        let model_builder_with_facts = model_inputs.iter().enumerate().try_fold(
            model_builder_with_properties,
            |current_builder, (i, tensor)| {
                let input_fact_result = model_ref.input_fact(i);
                println!("Original input fact for input {}: {:?}", i, input_fact_result);

                //let input_fact = i64::fact(tensor.shape()).into();
                
                let input_fact = match input_fact_result {
                    Ok(existing_fact) if existing_fact.shape.is_concrete() && existing_fact.datum_type().is_some() => {
                        println!("Using existing fact with resolved symbols: {:?}", existing_fact);
                        existing_fact.clone()
                        // let datum_type = existing_fact.datum_type().unwrap();
                        // let concrete_shape: TVec<TDim> = if tensor.shape().is_empty() {
                        //     tvec![TDim::Val(1)]
                        // } else {
                        //     tensor.shape()
                        //         .iter()
                        //         .map(|&d| TDim::Val(d as i64))
                        //         .collect()
                        // };
                        
                        // println!("Forcing concrete shape for input {}: {:?}", i, concrete_shape);
                        // InferenceFact::dt_shape(datum_type, concrete_shape)
                    },
                    _ => {
                        let mut datum_type = input_fact_result
                            .ok()
                            .and_then(|fact| fact.datum_type())
                            .unwrap_or_else(|| { tensor.datum_type()
                        });
                        let int64_dt = i64::datum_type();
                        let float32_dt = f32::datum_type();
                        let bool_dt = bool::datum_type();
                        let int32_dt = i32::datum_type();
                        datum_type = match datum_type {
                            dt if dt == int64_dt || dt == float32_dt || dt == bool_dt || dt == int32_dt => dt,
                            _ => int64_dt,
                        };

                        println!("Input {}: Using tensor shape: {:?}", i, tensor.shape());
                        let shape: TVec<TDim> = if !tensor.shape().is_empty() {
                            println!("Shape is available: {:?}", tensor.shape());
                            tensor.shape().iter().map(|&d| TDim::Val(d as i64)).collect()
                        } else {
                            match tensor.as_slice::<i64>() {
                                Ok(slice) if slice.len() > 0 => {
                                    println!("Shape is 1-D slice with length: {:?}", slice.len());
                                    tvec![TDim::Val(slice.len() as i64)]
                                }
                                _ => {
                                    println!("Shape is symbolic: [batch_size]");
                                    //let (batch_size, _sequence_length) = get_global_symbols();
                                    //tvec![TDim::Sym(batch_size.clone())]
                                    //tvec![]
                                    tvec![TDim::Val(1)]
                                }
                            }
                        };

                        println!("Creating concrete input fact with dtype: {:?} and shape: {:?}", datum_type, shape);
                        InferenceFact::dt_shape(datum_type, shape)
                        //InferenceFact::dt_shape(datum_type, tensor.shape())
                    }
                };
                current_builder.with_input_fact(i, input_fact).map_err(|e| {
                    println!("Failed to set input fact for index {}: {}", i, e);
                    e
                })
            },
        );

        println!("About to call into_typed()...");
        let m = model_builder_with_facts?.into_typed().map_err(|e| {
            println!("Failed to convert to typed model: {}", e);
            e
        })?;
 
        *transformed_model = Box::into_raw(Box::new(m));

        Ok(())
    })();

    handle_error(result)
}

pub type TractLlmTransformedModel = Graph<TypedFact, Box<dyn TypedOp>>;
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_optimized_llm(
    num_inputs: usize,
    input_shapefacts: *mut *mut c_void,
    input_datum_types: *mut *mut c_void,
    model: *mut *mut TractLlmInferenceModel,
    transformed_model: *mut *mut TractLlmTransformedModel,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        let shapefacts: Vec<Option<Vec<ShapeFact>>> = unsafe {
            std::slice::from_raw_parts(input_shapefacts, num_inputs)
                .iter()
                .map(|&ptr| {
                    if ptr.is_null() {
                        None
                    } else {
                        let shapefact_ref = &*(ptr as *mut Vec<ShapeFact>);
                        Some(shapefact_ref.clone())
                    }
                })
                .collect()
        };

        let mut datum_types: Vec<tract_core::prelude::DatumType> = unsafe {
            std::slice::from_raw_parts(input_datum_types, num_inputs)
                .iter()
                .map(|&ptr| {
                    if ptr.is_null() {
                        tract_core::prelude::DatumType::I64
                    } else {
                        let datum_ref = &*(ptr as *mut tract_core::prelude::DatumType);
                        datum_ref.clone()
                    }
                })
                .collect()
        };

        let model_ref: &mut TractLlmInferenceModel = {
            assert!(!model.is_null());
            let ptr = *model;
            assert!(!ptr.is_null());
            &mut *ptr
        };

        let mut model_builder = model_ref.clone();
        let original_input_facts: Vec<Option<tract_hir::infer::ShapeFactoid>> = (0..num_inputs)
            .map(|i| model_builder.input_fact(i).ok().map(|f| f.shape.clone()))
            .collect();

        let state = STATE.as_ref().ok_or_else(|| anyhow::anyhow!("STATE not initialized"))?;
        let seq_len = state.ids.len() as i64;
        let batch = 1i64;
        let mut solver = tract_core::prelude::SymbolValues::default();
        for sym_name in ["batch_size", "batch", "sequence_length", "sequence", "seq_len", "seq_length"] {
            let sym = model_builder.symbol_table.sym(sym_name);
            let val = if sym_name.contains("batch") { batch } else { seq_len };
            solver = solver.with(&sym, val);
        }

        // reset all node facts
        for node_id in 0..model_builder.nodes().len() {
            let node = model_builder.node(node_id);
            for i in 0..node.outputs.len() {
                model_builder.set_outlet_fact(
                    tract_core::internal::OutletId::new(node_id, i),
                    InferenceFact::default(),
                )?;
            }
        }

        for (i, shapefact_opt) in shapefacts.iter().enumerate() {
            match shapefact_opt {
                Some(shapefact_vec) => {
                    let shapefact = &shapefact_vec[0];
                    let dims: TVec<TDim> = shapefact
                        .to_tvec()
                        .iter()
                        .map(|d| d.eval(&solver))
                        .collect();
                    if datum_types[i] == TDim::datum_type() {
                        datum_types[i] = i64::datum_type();
                    }
                    let input_fact = InferenceFact::dt_shape(datum_types[i], dims);
                    model_builder = model_builder.with_input_fact(i, input_fact)?;
                },
                None => {
                    if let Some(shape) = &original_input_facts[i] {
                        match shape.concretize() {
                            Some(concrete_dims) => {
                                let resolved_shape: TVec<TDim> = concrete_dims
                                    .iter()
                                    .map(|d| d.eval(&solver))
                                    .collect();
                                model_builder = model_builder.with_input_fact(
                                    i,
                                    InferenceFact::dt_shape(datum_types[i], resolved_shape),
                                )?;
                            }
                            None => {
                                model_builder = model_builder.with_input_fact(
                                    i,
                                    InferenceFact::dt_shape(datum_types[i], shape.clone()),
                                )?;
                            }
                        }
                    }
                }
            }
        }

        let m = model_builder.into_optimized().map_err(|e| {
            eprintln!("Failed to convert to optimized model: {}", e);
            e
        })?;

        let model_arc = Arc::from_raw(*model);
        *model = Arc::into_raw(model_arc) as *mut _;
        *transformed_model = Box::into_raw(Box::new(m));

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_model_into_runnable_and_run_llm(
    inputs: *mut *mut c_void,
    num_inputs: usize,
    transformed_model: *mut *mut TractLlmTransformedModel,
    outputs: *mut *mut c_void,
    input_shapefacts: *mut *mut c_void,
    input_datum_types: *mut *mut c_void,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Start running llm");
        }        

        let model_inputs: SmallVec<[TValue; 4]> = unsafe {
            std::slice::from_raw_parts(inputs, num_inputs)
                .iter()
                .map(|&ptr| {
                    let tensor_ref = &*(ptr as *mut Tensor);
                    TValue::from(tensor_ref.clone())
                })
                .collect()
        };
        
        let typed: Box<TypedModel> = Box::from_raw(*transformed_model);
        *transformed_model = std::ptr::null_mut();
        for (ix, outlet) in typed.outputs.iter().enumerate() {
            let fact = typed.outlet_fact(*outlet)?;
            let shapefacts_vec: Vec<ShapeFact> = vec![fact.shape.clone()];
            *(input_shapefacts.add(ix)) = Box::into_raw(Box::new(shapefacts_vec)) as *mut c_void;
        }

        let model = typed.into_runnable()?;
        
        let output_vectors = model.run(model_inputs)?;
        for (i, output) in output_vectors.into_iter().enumerate() {
            let tensor = output.into_tensor();
            *(input_datum_types.add(i)) = Box::into_raw(Box::new(tensor.datum_type())) as *mut c_void;
            *(outputs.add(i)) = Box::into_raw(Box::new(tensor)) as *mut c_void;
        }

        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Finished running llm");
        }

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_generate_text_llm(
    inputs: *mut *mut c_void,
    num_inputs: usize,
    tokenizer_ptr: *mut c_void,
    outputs: *mut *mut c_void,
    num_outputs: usize,
    inference: *mut *mut c_char,
    next_token_id: *mut usize,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        let model_inputs: SmallVec<[TValue; 4]> = unsafe {
            std::slice::from_raw_parts(inputs, num_inputs)
                .iter()
                .map(|&ptr| {
                    let tensor_ref = &*(ptr as *mut Tensor);
                    TValue::from(tensor_ref.clone())
                })
                .collect()
        };

        let model_outputs: SmallVec<[TValue; 4]> = unsafe {
            std::slice::from_raw_parts(outputs, num_outputs)
                .iter()
                .map(|&ptr| {
                    let tensor_ref = &*(ptr as *mut Tensor);
                    TValue::from(tensor_ref.clone())
                })
                .collect()
        };

        let tokenizer = &*(tokenizer_ptr as *mut Tokenizer);

        let first_vec_u32: Vec<u32> = if let Ok(array) = model_inputs[0].to_array_view::<i64>() {
            array.iter().map(|x| *x as u32).collect()
        } else if let Ok(array) = model_inputs[0].to_array_view::<f32>() {
            array.iter().map(|x| *x as u32).collect()
        // } else if let Ok(array) = model_inputs[0].to_array_view::<TDim>() {
        //     array.iter().map(|x| x.to_i64().unwrap_or(0) as u32).collect()
        } else {
            return Err(anyhow::anyhow!(
                "Unsupported tensor type: {:?}",
                model_inputs[0].datum_type()
            ));
        };

        let logits = model_outputs[0].to_array_view::<f32>()?;
        let last_logits;
        let generated_text = match tokenizer.token_to_id("[MASK]") {
            Some(mask_id) => {
                let mask_pos = first_vec_u32
                    .iter()
                    .position(|&x| x == mask_id)
                    .ok_or_else(|| anyhow::anyhow!("Mask token not found"))?;
                last_logits = logits.slice(s![0, mask_pos, ..]);
                let word_id = last_logits.iter().zip(0..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap()).unwrap().1;
                tokenizer.id_to_token(word_id)
            }
            None => {
                last_logits = logits.slice(s![0, -1, ..]);
                
                // Top-k sampling
                let k = 10;
                let mut scored: Vec<(usize, f32)> = last_logits
                    .iter()
                    .cloned()
                    .enumerate()
                    .collect();

                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top_k = &scored[..k.min(scored.len())];
                *next_token_id = top_k
                    .choose(&mut thread_rng())
                    .map(|(idx, _)| *idx)
                    .unwrap();

                // Stop if model outputs <|endoftext|> token (50256 in GPT-2)
                let eos_token_id = tokenizer.get_vocab(true).get("<|endoftext|>").cloned().unwrap_or(50256);
                if *next_token_id == eos_token_id as usize {
                    return Err(anyhow::anyhow!("Reached EOS token: {}", eos_token_id));
                }

                tokenizer.decode(&first_vec_u32, true).ok()
            }
        };

        // Handle the Option and create a CString
        let re = regex::Regex::new(r"\s+")
            .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?;
        let clean_string = match generated_text {
            Some(generated_text) => re.replace_all(generated_text.trim(), " ").to_string(),
            None => "No generated_text found".to_string(),
        };
        let formatted_string = format!("Inference: {}", clean_string);
        let c_word = CString::new(formatted_string)?;
        *inference = c_word.into_raw(); // Pass the result back

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_update_input_values_llm(
    input_values: *mut *mut c_void,
    num_inputs: usize,
    next_token_id: usize,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        tract_free_llm_inputs(input_values, num_inputs);

        let state = STATE.as_mut().unwrap();
        state.ids.push(next_token_id as u32);
        state.attention_mask.push(1);

        if num_inputs == 3 {
            if let Some(&last) = state.third_vec.last() {
                state.third_vec.push(last + 1);
            } else {
                state.third_vec.push(0);
            }
        }

        let input_ids_tensor: Tensor = Array2::from_shape_vec(
            (1, state.ids.len()),
            state.ids.iter().map(|&x| x as i64).collect(),
        )?.into();

        let attention_mask_tensor: Tensor = Array2::from_shape_vec(
            (1, state.attention_mask.len()),
            state.attention_mask.iter().map(|&x| x as i64).collect(),
        )?.into();

        if num_inputs < 1 || num_inputs > 3 {
            return Err(anyhow::anyhow!("The input is not corrrect for an llm!"));
        }

        match num_inputs {
            1 => {
                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
            },
            2 => {
                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
                *(input_values.add(1)) = Box::into_raw(Box::new(attention_mask_tensor)) as *mut c_void;
            },
            _ => {
                *(input_values.add(0)) = Box::into_raw(Box::new(input_ids_tensor)) as *mut c_void;
                *(input_values.add(1)) = Box::into_raw(Box::new(attention_mask_tensor)) as *mut c_void;
                
                let third_tensor: Tensor = Array2::from_shape_vec(
                    (1, state.third_vec.len()),
                    state.third_vec.iter().map(|&x| x as i64).collect(),
                )?.into();

                *(input_values.add(2)) = Box::into_raw(Box::new(third_tensor)) as *mut c_void;
            }
        };

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_run_latest_models(
    model_path: *const c_char,
    tokenizer_ptr: *mut c_void,
    inference: *mut *mut c_char,
    num_tokens: usize,
    prompt: *const c_char,
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Start latest_model");
        }    
        
        let tokenizer = &*(tokenizer_ptr as *mut Tokenizer);

        let prompt_cstr = unsafe { CStr::from_ptr(prompt) };
        let prompt_str = prompt_cstr.to_str()?;
        let tokenizer_output_result = tokenizer.encode(prompt_str, true);
        let tokenizer_output = match tokenizer_output_result {
            Ok(output) => output,
            Err(_) => return Err(anyhow::anyhow!("Failed to encode text")),
        };

        let mut current_ids: Vec<u32> = tokenizer_output.get_ids().to_vec();
        let mut current_attention_mask: Vec<u32> = tokenizer_output.get_attention_mask().to_vec();
        let mut current_position_ids: Vec<u32> = (0..current_ids.len() as u32).collect();

        let model = {
            #[cfg(feature = "use_sys_time")]
            {
                let shape_input_ids = [1, current_ids.len()];
                let shape_attention_mask = [1, current_attention_mask.len()];
                let shape_position_ids = [1, current_position_ids.len()];
                let path = CStr::from_ptr(model_path).to_str()?;
                let model_dir = PathBuf::from_str(path)?;
                let decrypted = open_weights_file(Some(path))?;
                let weights_data = if decrypted.is_empty() {
                    None
                } else {
                    Some(decrypted)
                };

                tract_onnx::onnx().model_for_path(model_dir, weights_data.as_deref())?
                    .with_input_fact(0, i64::fact(shape_input_ids).into())?
                    .with_input_fact(1, i64::fact(shape_attention_mask).into())?
                    .with_input_fact(2, i64::fact(shape_position_ids).into())?
                    .into_typed()?
                    .into_runnable()?
            }

            #[cfg(not(feature = "use_sys_time"))]
            {
                let path = CStr::from_ptr(model_path).to_str()?;
                let model_dir = PathBuf::from_str(path)?;
                let decrypted = open_weights_file(Some(path))?;
                let weights_data = if decrypted.is_empty() {
                    None
                } else {
                    Some(decrypted)
                };

                tract_onnx::onnx().model_for_path(model_dir, weights_data.as_deref())?
                    .into_optimized()?
                    .into_runnable()?
            }
        };

        for _ in 0..num_tokens {
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
            let k = 3;
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
            let eos_token_id = tokenizer.get_vocab(true).get("<|endoftext|>").cloned().unwrap_or(50256);
            if next_token_id == eos_token_id {
                break;
            }

            current_ids.push(next_token_id);
            current_attention_mask.push(1);
            current_position_ids.push(current_position_ids.last().unwrap() + 1);

            #[cfg(not(feature = "use_sys_time"))]
            {
                print_memory("Before dropping outputs");
            }
            drop(outputs);
            #[cfg(not(feature = "use_sys_time"))]
            {
                print_memory("After dropping outputs");
            }
        }

        let generated_text = tokenizer.decode(&current_ids, true).map_err(|e| {
            anyhow::anyhow!("Failed to decode tokenizer output: {}", e)
        })?;
        
        // Handle the Option and create a CString
        let re = regex::Regex::new(r"\s+")
            .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?;
        let clean_string = re.replace_all(generated_text.trim(), " ").to_string();
        let formatted_string = format!("Inference: {}", clean_string);
        let c_word = CString::new(formatted_string)?;
        *inference = c_word.into_raw(); // Pass the result back

        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Before drop");
        }
        drop(model);
        drop(tokenizer_output);
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("After drop");
        }

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_run_albert(
    model_path: *const c_char,
    tokenizer_ptr: *mut c_void,
    inference: *mut *mut c_char,
    inference_model: *mut *mut TractLlmInferenceModel
) -> TRACT_RESULT {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Start albert");
        }
        
        let tokenizer = &*(tokenizer_ptr as *mut Tokenizer);

        let text = "Paris is the [MASK] of France.";
        let tokenizer_output_result = tokenizer.encode(text, true);
        let tokenizer_output = match tokenizer_output_result {
            Ok(output) => output,
            Err(_) => return Err(anyhow::anyhow!("Failed to encode text")),
        };

        let input_ids = tokenizer_output.get_ids();
        let attention_mask = tokenizer_output.get_attention_mask();
        let token_type_ids = tokenizer_output.get_type_ids();
        let length = input_ids.len();
        
        let model = {
            #[cfg(feature = "use_sys_time")]
            {
                let shape = [1, length];
                if inference_model.is_null() {
                    let path = CStr::from_ptr(model_path).to_str()?;
                    let model_dir = PathBuf::from_str(path)?;
                    tract_onnx::onnx().model_for_path(model_dir, None)?
                        .with_input_fact(0, i64::fact(shape).into())?
                        .with_input_fact(1, i64::fact(shape).into())?
                        .with_input_fact(2, i64::fact(shape).into())?
                        .into_typed()?
                        .into_runnable()?
                } else {
                    Box::from_raw(*inference_model)
                        .with_input_fact(0, i64::fact(shape).into())?
                        .with_input_fact(1, i64::fact(shape).into())?
                        .with_input_fact(2, i64::fact(shape).into())?
                        .into_typed()?
                        .into_runnable()?
                }
            }

            #[cfg(not(feature = "use_sys_time"))]
            {
                if inference_model.is_null() {
                    let path = CStr::from_ptr(model_path).to_str()?;
                    let model_dir = PathBuf::from_str(path)?;
                    tract_onnx::onnx().model_for_path(model_dir, None)?
                        .into_optimized()?
                        .into_runnable()?
                } else {
                    Box::from_raw(*inference_model)
                        .into_optimized()?
                        .into_runnable()?
                }
            }
        };

        let mask_pos = input_ids
            .iter()
            .position(|&x| x == tokenizer.token_to_id("[MASK]").unwrap())
            .ok_or_else(|| anyhow::anyhow!("Mask token not found"))?;

        let input_ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, length),
            input_ids.iter().map(|&x| x as i64).collect(),
        )?
        .into();
        let attention_mask_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, length),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )?
        .into();
        let token_type_ids_tensor: Tensor = tract_ndarray::Array2::from_shape_vec(
            (1, length),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )?
        .into();

        let outputs = model.run(tvec!(input_ids_tensor.into(), attention_mask_tensor.into(), token_type_ids_tensor.into()))?;
        let logits = outputs[0].to_array_view::<f32>()?;
        let logits = logits.slice(s![0, mask_pos, ..]);
        let word_id = logits.iter().zip(0..).max_by(|a, b| a.0.partial_cmp(b.0).unwrap()).unwrap().1;
        let word = tokenizer.id_to_token(word_id);

        // Handle the Option and create a CString
        let re = regex::Regex::new(r"\s+")
            .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?;
        let clean_string = match word {
            Some(word) => re.replace_all(word.trim(), " ").to_string(),
            None => "No word found".to_string(),
        };
        let formatted_string = format!("Inference: {}", clean_string);
        let c_word = CString::new(formatted_string)?;
        *inference = c_word.into_raw(); // Pass the result back

        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Before drop");
        }
        drop(model);
        drop(tokenizer_output);
        drop(outputs);
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("After drop");
        }

        Ok(())
    })();

    handle_error(result)
}

#[no_mangle]
pub unsafe extern "C" fn tract_run_gpt2(
    model_path: *const c_char,
    tokenizer_ptr: *mut c_void,
    inference: *mut *mut c_char,
    num_tokens: usize,
    prompt: *const c_char,
    inference_model: *mut *mut TractLlmInferenceModel
) -> TRACT_RESULT  {
    // Define the result to be returned
    let result = (|| -> Result<(), anyhow::Error> {
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Start gpt2");
        }
        
        let tokenizer = &*(tokenizer_ptr as *mut Tokenizer);

        let prompt_cstr = unsafe { CStr::from_ptr(prompt) };
        let prompt_str = prompt_cstr.to_str()?;
        let tokenizer_output_result = tokenizer.encode(prompt_str, true);
        let tokenizer_output = match tokenizer_output_result {
            Ok(output) => output,
            Err(_) => return Err(anyhow::anyhow!("Failed to encode text")),
        };

        let mut current_ids: Vec<u32> = tokenizer_output.get_ids().to_vec();
        let mut current_attention_mask: Vec<u32> = tokenizer_output.get_attention_mask().to_vec();

        let model = {
            #[cfg(feature = "use_sys_time")]
            {
                let shape_input_ids = [1, current_ids.len()];
                let shape_attention_mask = [1, current_attention_mask.len()];
                if inference_model.is_null() {
                    let path = CStr::from_ptr(model_path).to_str()?;
                    let model_dir = PathBuf::from_str(path)?;
                    tract_onnx::onnx().model_for_path(model_dir, None)?
                        .with_input_fact(0, i64::fact(shape_input_ids).into())?
                        .with_input_fact(1, i64::fact(shape_attention_mask).into())?
                        .into_typed()?
                        .into_runnable()?
                } else {
                    Box::from_raw(*inference_model)
                        .with_input_fact(0, i64::fact(shape_input_ids).into())?
                        .with_input_fact(1, i64::fact(shape_attention_mask).into())?
                        .into_typed()?
                        .into_runnable()?
                }
            }

            #[cfg(not(feature = "use_sys_time"))]
            {
                if inference_model.is_null() {
                    let path = CStr::from_ptr(model_path).to_str()?;
                    let model_dir = PathBuf::from_str(path)?;
                    tract_onnx::onnx().model_for_path(model_dir, None)?
                        .into_optimized()?
                        .into_runnable()?
                } else {
                    Box::from_raw(*inference_model)
                        .into_optimized()?
                        .into_runnable()?
                }
            }
        };

        for _ in 0..num_tokens {
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

            #[cfg(not(feature = "use_sys_time"))]
            {
                print_memory("Before dropping outputs");
            }
            drop(outputs);
            #[cfg(not(feature = "use_sys_time"))]
            {
                print_memory("After dropping outputs");
            }
        }

        let generated_text = tokenizer.decode(&current_ids, true).map_err(|e| {
            anyhow::anyhow!("Failed to decode tokenizer output: {}", e)
        })?;
        
        // Handle the Option and create a CString
        let re = regex::Regex::new(r"\s+")
            .map_err(|e| anyhow::anyhow!("Failed to compile regex: {}", e))?;
        let clean_string = re.replace_all(generated_text.trim(), " ").to_string();
        let formatted_string = format!("Inference: {}", clean_string);
        let c_word = CString::new(formatted_string)?;
        *inference = c_word.into_raw(); // Pass the result back
        
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("Before drop");
        }
        drop(model);
        drop(tokenizer_output);
        #[cfg(not(feature = "use_sys_time"))]
        {
            print_memory("After drop");
        }

        Ok(())
    })();

    handle_error(result)
}

/// Creates an instance of an ONNX framework and parser that can be used to load models.
///
/// The returned object should be destroyed with `tract_nnef_destroy` once the model
/// has been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_create(onnx: *mut *mut TractOnnx) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(onnx);
        *onnx = Box::into_raw(Box::new(TractOnnx(tract_rs::onnx()?)));
        Ok(())
    })
}

/// Destroy the NNEF parser. It is safe to detroy the NNEF parser once the model had been loaded.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_destroy(onnx: *mut *mut TractOnnx) -> TRACT_RESULT {
    release!(onnx)
}

/// Parse and load an ONNX model as a tract InferenceModel.
/// println!("cargo:rerun-if-changed=tract.h");
/// `path` is a null-terminated utf-8 string pointer. It must point to a `.onnx` model file.
#[no_mangle]
pub unsafe extern "C" fn tract_onnx_model_for_path_cnn(
    onnx: *const TractOnnx,
    path: *const c_char,
    model: *mut *mut TractInferenceModel
) -> TRACT_RESULT {
    wrap(|| unsafe {
        // Inside tract_onnx_model_for_path function
        check_not_null!(onnx, path, model);

        *model = std::ptr::null_mut();
        let path = CStr::from_ptr(path).to_str()?;
        let m = Arc::new(TractInferenceModel((*onnx).0.model_for_path(path)?));
        *model = Arc::into_raw(m) as *mut _;
        Ok(())
    })
}

// INFERENCE MODEL
pub struct TractInferenceModel(tract_rs::InferenceModel);

/// Query an InferenceModel input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_count(
    model: *const TractInferenceModel,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an InferenceModel output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_count(
    model: *const TractInferenceModel,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

/// Query the name of a model input.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_name(
    model: *const TractInferenceModel,
    input: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(&*m.input_name(input)?)?.into_raw();
        Ok(())
    })
}

/// Query the name of a model output.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_name(
    model: *const TractInferenceModel,
    output: usize,
    name: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(&*m.output_name(output)?)?.into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_input_fact(
    model: *const TractInferenceModel,
    input_id: usize,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.input_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Set an input fact of an InferenceModel.
///
/// The `fact` argument is only borrowed by this function, it still must be destroyed.
/// `fact` can be set to NULL to erase the current output fact of the model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_input_fact(
    model: *mut TractInferenceModel,
    input_id: usize,
    fact: *const TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_input_fact(input_id, f)?;
        Ok(())
    })
}

/// Change the model outputs nodes (by name).
///
/// `names` is an array containing `len` pointers to null terminated strings.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_output_names(
    model: *mut TractInferenceModel,
    len: usize,
    names: *const *const c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names, *names);
        let node_names = (0..len)
            .map(|i| Ok(CStr::from_ptr(*names.add(i)).to_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        (*model).0.set_output_names(&node_names)?;
        Ok(())
    })
}

/// Query an output fact for an InferenceModel.
///
/// The return model must be freed using `tract_inference_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_output_fact(
    model: *const TractInferenceModel,
    output_id: usize,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.output_fact(output_id)?;
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Set an output fact of an InferenceModel.
///
/// The `fact` argument is only borrowed by this function, it still must be destroyed.
/// `fact` can be set to NULL to erase the current output fact of the model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_set_output_fact(
    model: *mut TractInferenceModel,
    output_id: usize,
    fact: *const TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        let f = fact.as_ref().map(|f| &f.0).cloned().unwrap_or_default();
        (*model).0.set_output_fact(output_id, f)?;
        Ok(())
    })
}

/// Analyse an InferencedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_analyse(
    model: *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.analyse()?;
        Ok(())
    })
}

/// Convenience function to obtain an optimized TypedModel from an InferenceModel.
///
/// This function takes ownership of the InferenceModel `model` whether it succeeds
/// or not. `tract_inference_model_destroy` must not be used on `model`.
///
/// On the other hand, caller will be owning the newly created `optimized` model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_optimized(
    model: *mut *mut TractInferenceModel,
    optimized: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model, optimized);
        *optimized = std::ptr::null_mut();
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        let result = m.0.into_optimized()?;
        *optimized = Box::into_raw(Box::new(TractModel(result))) as _;
        Ok(())
    })
}

/// Function to release the inference_model
#[no_mangle]
pub unsafe extern "C" fn tract_cnn_inference_model_release(
    model: *mut *mut TractInferenceModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model);
        let model_ptr = *model;
        let _ = Arc::from_raw(model_ptr);
        *model = std::ptr::null_mut();
        Ok(())
    })
}

/// Transform a fully analysed InferenceModel to a TypedModel.
///
/// This function takes ownership of the InferenceModel `model` whether it succeeds
/// or not. `tract_inference_model_destroy` must not be used on `model`.
///
/// On the other hand, caller will be owning the newly created `optimized` model.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_model_into_typed(
    model: *mut *mut TractInferenceModel,
    typed: *mut *mut TractModel,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model, typed);
        let model_arc = Arc::from_raw(*model);
        let cloned_model_arc = model_arc.clone();
        let result = cloned_model_arc.0.clone().into_typed();
        *model = Arc::into_raw(model_arc) as *mut _;

        match result {
            Ok(typed_model) => {
                *typed = Box::into_raw(Box::new(TractModel(typed_model))) as *mut _;
                Ok(())
            }
            Err(e) => {
                Err(e)
            }
        }
    })
}

// TYPED MODEL

pub struct TractModel(tract_rs::Model);

/// Query an InferenceModel input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_count(
    model: *const TractModel,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an InferenceModel output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_count(
    model: *const TractModel,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

/// Query the name of a model input.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_name(
    model: *const TractModel,
    input: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(m.input_name(input)?)?.into_raw();
        Ok(())
    })
}

/// Query the input fact of a model.
///
/// Thre returned fact must be freed with tract_fact_destroy.
#[no_mangle]
pub unsafe extern "C" fn tract_model_input_fact(
    model: *const TractModel,
    input_id: usize,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.input_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Query the name of a model output.
///
/// The returned name must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_name(
    model: *const TractModel,
    output: usize,
    name: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name);
        *name = std::ptr::null_mut();
        let m = &(*model).0;
        *name = CString::new(m.output_name(output)?)?.into_raw();
        Ok(())
    })
}

/// Query the output fact of a model.
///
/// Thre returned fact must be freed with tract_fact_destroy.
#[no_mangle]
pub unsafe extern "C" fn tract_model_output_fact(
    model: *const TractModel,
    input_id: usize,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, fact);
        *fact = std::ptr::null_mut();
        let f = (*model).0.output_fact(input_id)?;
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Change the model outputs nodes (by name).
///
/// `names` is an array containing `len` pointers to null terminated strings.
#[no_mangle]
pub unsafe extern "C" fn tract_model_set_output_names(
    model: *mut TractModel,
    len: usize,
    names: *const *const c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names, *names);
        let node_names = (0..len)
            .map(|i| Ok(CStr::from_ptr(*names.add(i)).to_str()?.to_owned()))
            .collect::<Result<Vec<_>>>()?;
        (*model).0.set_output_names(&node_names)
    })
}

/// Give value one or more symbols used in the model.
///
/// * symbols is an array of `nb_symbols` pointers to null-terminated UTF-8 string for the symbols
/// names to substitue
/// * values is an array of `nb_symbols` integer values
#[no_mangle]
pub unsafe extern "C" fn tract_model_concretize_symbols(
    model: *mut TractModel,
    nb_symbols: usize,
    symbols: *const *const i8,
    values: *const i64,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, symbols, values);
        let model = &mut (*model).0;
        let mut table = vec![];
        for i in 0..nb_symbols {
            let name = CStr::from_ptr(*symbols.add(i))
                .to_str()
                .with_context(|| {
                    format!("failed to parse symbol name for {i}th symbol (not utf8)")
                })?
                .to_owned();
            table.push((name, *values.add(i)));
        }
        model.concretize_symbols(table)
    })
}

/// Pulsify the model
///
/// * stream_symbol is the name of the stream symbol
/// * pulse expression is a dim to use as the pulse size (like "8", "P" or "3*p").
#[no_mangle]
pub unsafe extern "C" fn tract_model_pulse_simple(
    model: *mut *mut TractModel,
    stream_symbol: *const i8,
    pulse_expr: *const i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, *model, stream_symbol, pulse_expr);
        let model = &mut (**model).0;
        let stream_sym = CStr::from_ptr(stream_symbol)
            .to_str()
            .context("failed to parse stream symbol name (not utf8)")?;
        let pulse_dim = CStr::from_ptr(pulse_expr)
            .to_str()
            .context("failed to parse stream symbol name (not utf8)")?;
        model.pulse(stream_sym, pulse_dim)
    })
}

/// Apply a transform to the model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_transform(
    model: *mut TractModel,
    transform: *const i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, transform);
        let t = CStr::from_ptr(transform)
            .to_str()
            .context("failed to parse transform name (not utf8)")?;
        (*model).0.transform(t)
    })
}

/// Declutter a TypedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_model_declutter(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.declutter()
    })
}

/// Optimize a TypedModel in-place.
#[no_mangle]
pub unsafe extern "C" fn tract_model_optimize(model: *mut TractModel) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model);
        (*model).0.optimize()
    })
}

/// Perform a profile of the model using the provided inputs.
#[no_mangle]
pub unsafe extern "C" fn tract_model_profile_json(
    model: *mut TractModel,
    inputs: *mut *mut TractValue,
    json: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, json);
        let input: Option<Vec<Value>> = if !inputs.is_null() {
            let input_len = (*model).0.input_count()?;
            Some(
                std::slice::from_raw_parts(inputs, input_len)
                    .iter()
                    .map(|tv| (**tv).0.clone())
                    .collect(),
            )
        } else {
            None
        };
        let profile = (*model).0.profile_json(input)?;
        *json = CString::new(profile)?.into_raw();
        Ok(())
    })
}

/// Convert a TypedModel into a TypedRunnableModel.
///
/// This function transfers ownership of the `model` argument to the newly-created `runnable` model.
///
/// Runnable are reference counted. When done, it should be released with `tract_runnable_release`.
#[no_mangle]
pub unsafe extern "C" fn tract_model_into_runnable(
    model: *mut *mut TractModel,
    runnable: *mut *mut TractRunnable,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, runnable);
        let m = Box::from_raw(*model);
        *model = std::ptr::null_mut();
        *runnable = Box::into_raw(Box::new(TractRunnable(m.0.into_runnable()?))) as _;
        Ok(())
    })
}

/// Query the number of properties in a model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property_count(
    model: *const TractModel,
    count: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, count);
        *count = (*model).0.property_keys()?.len();
        Ok(())
    })
}

/// Query the properties names of a model.
///
/// The "names" array should be big enough to fit `tract_model_property_count` string pointers.
///
/// Each name will have to be freed using `tract_free_cstring`.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property_names(
    model: *const TractModel,
    names: *mut *mut i8,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, names);
        for (ix, name) in (*model).0.property_keys()?.iter().enumerate() {
            *names.add(ix) = CString::new(&**name)?.into_raw();
        }
        Ok(())
    })
}

/// Query a property value in a model.
#[no_mangle]
pub unsafe extern "C" fn tract_model_property(
    model: *const TractModel,
    name: *const i8,
    value: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, name, value);
        let name = CStr::from_ptr(name)
            .to_str()
            .context("failed to parse property name (not utf8)")?
            .to_owned();
        let v = (*model).0.property(name).context("Property not found")?;
        *value = Box::into_raw(Box::new(TractValue(v)));
        Ok(())
    })
}

/// Destroy a TypedModel.
#[no_mangle]
pub unsafe extern "C" fn tract_model_destroy(model: *mut *mut TractModel) -> TRACT_RESULT {
    release!(model)
}

// RUNNABLE MODEL
pub struct TractRunnable(tract_rs::Runnable);

/// Spawn a session state from a runnable model.
///
/// This function does not take ownership of the `runnable` object, it can be used again to spawn
/// other state instances. The runnable object is internally reference counted, it will be
/// kept alive as long as any associated `State` exists (or as long as the `runnable` is not
/// explicitely release with `tract_runnable_release`).
///
/// `state` is a newly-created object. It should ultimately be detroyed with `tract_state_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_spawn_state(
    runnable: *mut TractRunnable,
    state: *mut *mut TractState,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(runnable, state);
        *state = std::ptr::null_mut();
        let s = (*runnable).0.spawn_state()?;
        *state = Box::into_raw(Box::new(TractState(s)));
        Ok(())
    })
}

/// Convenience function to run a stateless model.
///
/// `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
/// to the number of inputs of the models. The function does not take ownership of the input
/// values.
/// `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
/// with pointers to outputs values. These values are under the responsiblity of the caller, it
/// will have to release them with `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_run(
    runnable: *mut TractRunnable,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(runnable);
        let mut s = (*runnable).0.spawn_state()?;
        state_run(&mut s, inputs, outputs)
    })
}

/// Query a Runnable input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_input_count(
    model: *const TractRunnable,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, inputs);
        let model = &(*model).0;
        *inputs = model.input_count()?;
        Ok(())
    })
}

/// Query an Runnable output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_runnable_output_count(
    model: *const TractRunnable,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, outputs);
        let model = &(*model).0;
        *outputs = model.output_count()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_runnable_release(runnable: *mut *mut TractRunnable) -> TRACT_RESULT {
    release!(runnable)
}

// VALUE
pub struct TractValue(tract_rs::Value);

/// Create a TractValue (aka tensor) from caller data and metadata.
///
/// This call copies the data into tract space. All the pointers only need to be alive for the
/// duration of the call.
///
/// rank is the number of dimensions of the tensor (i.e. the length of the shape vector).
///
/// The returned value must be destroyed by `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_value_from_bytes(
    datum_type: DatumType,
    rank: usize,
    shape: *const usize,
    data: *mut c_void,
    value: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(value);
        *value = std::ptr::null_mut();
        let shape = std::slice::from_raw_parts(shape, rank);
        let len = shape.iter().product::<usize>();
        let data = std::slice::from_raw_parts(data as *const u8, len * datum_type.size_of());
        let it = Value::from_bytes(datum_type, shape, data)?;
        *value = Box::into_raw(Box::new(TractValue(it)));
        Ok(())
    })
}

/// Destroy a value.
#[no_mangle]
pub unsafe extern "C" fn tract_cnn_value_destroy(value: *mut *mut TractValue) -> TRACT_RESULT {
    release!(value)
}

/// Inspect part of a value. Except `value`, all argument pointers can be null if only some specific bits
/// are required.
#[no_mangle]
pub unsafe extern "C" fn tract_value_as_bytes(
    value: *mut TractValue,
    datum_type: *mut DatumType,
    rank: *mut usize,
    shape: *mut *const usize,
    data: *mut *const c_void,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(value);
        let value = &(*value).0;
        let bits = value.as_bytes()?;
        if !datum_type.is_null() {
            *datum_type = bits.0;
        }
        if !rank.is_null() {
            *rank = bits.1.len();
        }
        if !shape.is_null() {
            *shape = bits.1.as_ptr();
        }
        if !data.is_null() {
            *data = bits.2.as_ptr() as _;
        }
        Ok(())
    })
}

// STATE
pub struct TractState(tract_rs::State);

/// Run a turn on a model state
///
/// `inputs` is a pointer to an pre-existing array of input TractValue. Its length *must* be equal
/// to the number of inputs of the models. The function does not take ownership of the input
/// values.
/// `outputs` is a pointer to a pre-existing array of TractValue pointers that will be overwritten
/// with pointers to outputs values. These values are under the responsiblity of the caller, it
/// will have to release them with `tract_value_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_state_run(
    state: *mut TractState,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(state, inputs, outputs);
        state_run(&mut (*state).0, inputs, outputs)
    })
}

/// Query a State input counts.
#[no_mangle]
pub unsafe extern "C" fn tract_state_input_count(
    state: *const TractState,
    inputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(state, inputs);
        let state = &(*state).0;
        *inputs = state.input_count()?;
        Ok(())
    })
}

/// Query an State output counts.
#[no_mangle]
pub unsafe extern "C" fn tract_state_output_count(
    state: *const TractState,
    outputs: *mut usize,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(state, outputs);
        let state = &(*state).0;
        *outputs = state.output_count()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_state_destroy(state: *mut *mut TractState) -> TRACT_RESULT {
    release!(state)
}

// FACT
pub struct TractFact(tract_rs::Fact);

/// Parse a fact specification string into an Fact.
///
/// The returned fact must be free with `tract_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_fact_parse(
    model: *mut TractModel,
    spec: *const c_char,
    fact: *mut *mut TractFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, spec, fact);
        let spec = CStr::from_ptr(spec).to_str()?;
        let f: tract_rs::Fact = spec.as_fact(&mut (*model).0)?.as_ref().clone();
        *fact = Box::into_raw(Box::new(TractFact(f)));
        Ok(())
    })
}

/// Write a fact as its specification string.
///
/// The returned string must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_fact_dump(
    fact: *const TractFact,
    spec: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact, spec);
        *spec = CString::new(format!("{}", (*fact).0))?.into_raw();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn tract_fact_destroy(fact: *mut *mut TractFact) -> TRACT_RESULT {
    release!(fact)
}

// INFERENCE FACT
pub struct TractInferenceFact(tract_rs::InferenceFact);

/// Parse a fact specification string into an InferenceFact.
///
/// The returned fact must be free with `tract_inference_fact_destroy`.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_parse(
    model: *mut TractInferenceModel,
    spec: *const c_char,
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(model, spec, fact);
        let spec = CStr::from_ptr(spec).to_str()?;
        let f: tract_rs::InferenceFact = spec.as_fact(&mut (*model).0)?.as_ref().clone();
        *fact = Box::into_raw(Box::new(TractInferenceFact(f)));
        Ok(())
    })
}

/// Creates an empty inference fact.
///
/// The returned fact must be freed by the caller using tract_inference_fact_destroy
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_empty(
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact);
        *fact = Box::into_raw(Box::new(TractInferenceFact(Default::default())));
        Ok(())
    })
}

/// Write an inference fact as its specification string.
///
/// The returned string must be freed by the caller using tract_free_cstring.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_dump(
    fact: *const TractInferenceFact,
    spec: *mut *mut c_char,
) -> TRACT_RESULT {
    wrap(|| unsafe {
        check_not_null!(fact, spec);
        *spec = CString::new(format!("{}", (*fact).0))?.into_raw();
        Ok(())
    })
}

/// Destroy a fact.
#[no_mangle]
pub unsafe extern "C" fn tract_inference_fact_destroy(
    fact: *mut *mut TractInferenceFact,
) -> TRACT_RESULT {
    release!(fact)
}

// MISC

// HELPERS

unsafe fn state_run(
    state: &mut State,
    inputs: *mut *mut TractValue,
    outputs: *mut *mut TractValue,
) -> Result<()> {
    let values: Vec<_> = std::slice::from_raw_parts(inputs, state.input_count()?)
        .iter()
        .map(|tv| (**tv).0.clone())
        .collect();
    let values = state.run(values)?;
    for (i, value) in values.into_iter().enumerate() {
        *(outputs.add(i)) = Box::into_raw(Box::new(TractValue(value)))
    }
    Ok(())
}
