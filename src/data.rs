//! Dataset configuration and loading for JSON-driven training runs.

use flate2::read::GzDecoder;
use serde::Deserialize;
use std::{fs, io::Read, path::{Path, PathBuf}};
use wasmtime::{Engine, Instance, Memory, Module, Store, TypedFunc};

#[derive(Debug, Deserialize)]
pub struct TrainingData {
    pub name: String,
    #[serde(default)] pub loading_functions: Vec<LoadingFunction>,
    pub training: TrainingSource,
    #[serde(default)] pub validation: Option<TrainingSource>,
    #[serde(default)] pub test: Option<TrainingSource>,
    #[serde(default)] pub validation_samples: usize,
    #[serde(default)] pub target_accuracy: Option<f32>,
    #[serde(default = "one")] pub max_epochs: usize,
    #[serde(default)] pub max_samples: Option<usize>,
    #[serde(default = "default_batch_size")] pub batch_size: usize,
    #[serde(default)] pub use_cuda: bool,
}

#[derive(Debug, Deserialize)]
pub struct LoadingFunction { pub name: String, pub wasm_hex: String }

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingSource {
    Inline { inputs: Vec<Vec<f32>>, outputs: Vec<Vec<f32>> },
    Files { inputs: FileTrainingSet, outputs: FileTrainingSet, input_width: usize, output_width: usize },
    Mnist { images_path: PathBuf, labels_path: PathBuf },
}

#[derive(Debug, Deserialize)]
pub struct FileTrainingSet { pub file_path: PathBuf, pub loading_function: String }

fn one() -> usize { 1 }
fn default_batch_size() -> usize { 32 }

pub fn read_config(training_file: &str) -> Result<TrainingData, String> {
    serde_json::from_str(&fs::read_to_string(training_file).map_err(|error| error.to_string())?).map_err(|error| format!("invalid training JSON: {error}"))
}

pub fn test_pairs_from_json(training_file: &str) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let config = read_config(training_file)?;
    let test = config.test.as_ref().ok_or("training JSON needs a test source")?;
    training_source_sets(&config.loading_functions, test)
}

/// Splits validation data off the end of the training set unless an explicit source is configured.
pub fn validation_sets(config: &TrainingData, inputs: &mut Vec<Vec<f32>>, outputs: &mut Vec<Vec<f32>>) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    if let Some(source) = &config.validation { return training_source_sets(&config.loading_functions, source); }
    if config.validation_samples == 0 || config.validation_samples >= inputs.len() { return Ok((inputs.clone(), outputs.clone())); }
    let validation_inputs = inputs.split_off(inputs.len() - config.validation_samples);
    let validation_outputs = outputs.split_off(outputs.len() - config.validation_samples);
    Ok((validation_inputs, validation_outputs))
}

pub fn training_sets(config: &TrainingData) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    training_source_sets(&config.loading_functions, &config.training)
}

fn training_source_sets(loading_functions: &[LoadingFunction], source: &TrainingSource) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    match source {
        TrainingSource::Inline { inputs, outputs } => Ok((inputs.clone(), outputs.clone())),
        TrainingSource::Mnist { images_path, labels_path } => load_mnist(images_path, labels_path),
        TrainingSource::Files { inputs, outputs, input_width, output_width } => {
            let find = |name: &str| loading_functions.iter().find(|function| function.name == name).ok_or_else(|| format!("unknown loading function: {name}"));
            let x = wasm_load(find(&inputs.loading_function)?, &fs::read(&inputs.file_path).map_err(|error| error.to_string())?)?;
            let y = wasm_load(find(&outputs.loading_function)?, &fs::read(&outputs.file_path).map_err(|error| error.to_string())?)?;
            Ok((chunks(x, *input_width)?, chunks(y, *output_width)?))
        }
    }
}

// Loader ABI: export memory, alloc(i32)->i32, and load(i32, i32)->i64. load returns (count << 32) | pointer; output is little-endian f32 values.
fn wasm_load(loader: &LoadingFunction, input: &[u8]) -> Result<Vec<f32>, String> {
    let engine = Engine::default();
    let module = Module::new(&engine, hex(&loader.wasm_hex)?).map_err(|error| format!("compiling {}: {error}", loader.name))?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).map_err(|error| error.to_string())?;
    let memory: Memory = instance.get_memory(&mut store, "memory").ok_or("loader must export memory")?;
    let alloc: TypedFunc<i32, i32> = instance.get_typed_func(&mut store, "alloc").map_err(|_| "loader must export alloc(i32)->i32")?;
    let load: TypedFunc<(i32, i32), i64> = instance.get_typed_func(&mut store, "load").map_err(|_| "loader must export load(i32,i32)->i64")?;
    let pointer = alloc.call(&mut store, input.len() as i32).map_err(|error| error.to_string())?;
    memory.write(&mut store, pointer as usize, input).map_err(|error| error.to_string())?;
    let packed = load.call(&mut store, (pointer, input.len() as i32)).map_err(|error| error.to_string())? as u64;
    let count = (packed >> 32) as usize;
    let mut bytes = vec![0; count.checked_mul(4).ok_or("WASM output is too large")?];
    memory.read(&store, packed as u32 as usize, &mut bytes).map_err(|error| error.to_string())?;
    Ok(bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect())
}

fn load_mnist(images_path: &Path, labels_path: &Path) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
    let images = gzip(images_path)?;
    let labels = gzip(labels_path)?;
    if images.len() < 16 || labels.len() < 8 || images[..4] != [0, 0, 8, 3] || labels[..4] != [0, 0, 8, 1] { return Err("invalid MNIST IDX header".into()); }
    let count = be(&images[4..8])? as usize;
    let pixels = be(&images[8..12])? as usize * be(&images[12..16])? as usize;
    if count != be(&labels[4..8])? as usize || images.len() != 16 + count * pixels || labels.len() != 8 + count { return Err("inconsistent MNIST data".into()); }
    let x = images[16..].chunks_exact(pixels).map(|image| image.iter().map(|pixel| *pixel as f32 / 255.0).collect()).collect();
    let y = labels[8..].iter().map(|&label| {
        if label > 9 { return Err("invalid MNIST label".into()); }
        let mut one_hot = vec![0.0; 10];
        one_hot[label as usize] = 1.0;
        Ok(one_hot)
    }).collect::<Result<_, String>>()?;
    Ok((x, y))
}

fn gzip(path: &Path) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    GzDecoder::new(fs::File::open(path).map_err(|error| error.to_string())?).read_to_end(&mut result).map_err(|error| error.to_string())?;
    Ok(result)
}

fn be(value: &[u8]) -> Result<u32, String> { Ok(u32::from_be_bytes(value.try_into().map_err(|_| "bad IDX integer")?)) }

fn chunks(values: Vec<f32>, width: usize) -> Result<Vec<Vec<f32>>, String> {
    if width == 0 || values.len() % width != 0 { return Err("loader output does not fit sample width".into()); }
    Ok(values.chunks_exact(width).map(Vec::from).collect())
}

fn hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 { return Err("WASM hex must be even-length".into()); }
    (0..value.len()).step_by(2).map(|index| u8::from_str_radix(&value[index..index + 2], 16).map_err(|_| "invalid WASM hex".into())).collect()
}
