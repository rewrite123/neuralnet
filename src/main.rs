
#[cfg(feature = "gpu")]
mod gpu;
mod gguf;
mod growth;
mod model;
mod safetensors;
mod tokenizer;
mod transformer;
mod training;
mod learn_functions;
mod data;

use clap::{Parser, Subcommand, ValueEnum};
use rand::seq::SliceRandom;
use std::{error::Error, ffi::OsString, fs, io::{self, Write}, path::{Path, PathBuf}, sync::{atomic::{AtomicBool, Ordering}, Arc}};

#[derive(Parser)]
#[command(name = "neuralnet", about = "Create, train, and inspect dense neural-network models.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    New {
        #[arg(long, value_enum, default_value_t = Architecture::Dense)]
        architecture: Architecture,
        #[arg(long, default_value = "784,128,10")]
        layers: String,
        #[arg(long)]
        hidden: Option<usize>,
        #[arg(long)]
        slots: Option<usize>,
        #[arg(long)]
        blocks: Option<usize>,
        #[arg(long, default_value_t = 256)]
        vocab: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long)]
        ff_hidden: Option<usize>,
        #[arg(long, default_value_t = 128)]
        max_sequence: usize,
        #[arg(short, long, default_value = "models/model.gguf")]
        output: PathBuf,
    },
    /// Trains a transformer language model on a plain-text corpus.
    TrainText {
        #[arg(long)] model: PathBuf,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long)] text: PathBuf,
        #[arg(long, default_value = "data/gpt2/vocab.json")] vocab: PathBuf,
        #[arg(long, default_value = "data/gpt2/merges.txt")] merges: PathBuf,
        #[arg(long, default_value_t = 64)] sequence: usize,
        #[arg(long, default_value_t = 1)] epochs: usize,
        #[arg(long, default_value_t = 0.0003)] learning_rate: f32,
        #[arg(long)] max_sequences: Option<usize>,
        #[arg(long, default_value_t = 64)] log_every: usize,
        #[arg(long, default_value_t = 0.1)] validation_fraction: f32,
        #[arg(long, value_enum)] growth: Option<growth::GrowthStrategy>,
        #[arg(long, default_value_t = 256)] growth_max_units: usize,
        #[arg(long, default_value_t = 0.01)] growth_trigger: f32,
        #[arg(long, default_value_t = 2)] growth_patience: usize,
        #[arg(long, default_value_t = 0.20)] shrink_trigger: f32,
        #[arg(long, default_value_t = 2)] shrink_patience: usize,
        #[arg(long, default_value_t = 2.0)] growth_new_layer_ratio: f32,
        #[arg(long, default_value_t = 4)] growth_max_blocks: usize,
        #[arg(long, default_value_t = 3072)] growth_max_ff: usize,
        #[arg(long, default_value_t = 3)] growth_interval: usize,
        #[arg(long)] cuda: bool,
    },
    /// Greedily continues a prompt with a transformer language model.
    Generate {
        #[arg(long)] model: PathBuf,
        #[arg(long)] prompt: String,
        #[arg(long, default_value_t = 20)] tokens: usize,
        #[arg(long, default_value = "data/gpt2/vocab.json")] vocab: PathBuf,
        #[arg(long, default_value = "data/gpt2/merges.txt")] merges: PathBuf,
        #[arg(long, default_value_t = 0.0)] temperature: f32,
        #[arg(long, default_value_t = 40)] top_k: usize,
        #[arg(long)] cuda: bool,
    },
    /// Copies a pretrained matrix from a safetensors file into a model's embedding, position, or head layer.
    ImportEmbeddings {
        #[arg(long)] model: PathBuf,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long)] safetensors: PathBuf,
        #[arg(long, default_value = "wte.weight")] tensor: String,
        #[arg(long, value_enum, default_value_t = ImportTarget::Token)] target: ImportTarget,
    },
    Train {
        #[arg(long, value_enum, default_value_t = Architecture::Dense)]
        architecture: Architecture,
        #[arg(short, long, default_value = "training/mnist.json")]
        config: PathBuf,
        #[arg(short, long, default_value = "models/mnist.gguf")]
        output: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value = "784,128,10")]
        layers: String,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long, value_enum, default_value_t = Optimizer::Momentum)]
        optimizer: Optimizer,
        #[arg(long, default_value_t = 0.9)]
        momentum: f32,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        epochs: Option<usize>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        max_samples: Option<usize>,
    },
    AddLayer {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long)]
        layer_size: usize,
        #[arg(long)]
        insert_at: usize,
        #[arg(long, value_enum, default_value_t = LayerMethod::Passthrough)]
        method: LayerMethod,
        #[arg(long)]
        copy_from: Option<usize>,
        #[arg(long)]
        values_file: Option<PathBuf>,
    },
    AddNeurons {
        #[arg(long)] model: PathBuf,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long)] layer: usize,
        #[arg(long)] insert_at: usize,
        #[arg(long)] count: usize,
        #[arg(long, value_enum, default_value_t = NeuronMethod::Passthrough)] method: NeuronMethod,
    },
    RemoveNeurons {
        #[arg(long)] model: PathBuf,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long)] layer: usize,
        #[arg(long)] indexes: String,
    },
    RemoveLayer {
        #[arg(long)] model: PathBuf,
        #[arg(short, long)] output: Option<PathBuf>,
        #[arg(long)] layer: usize,
    },
    Evaluate {
        #[arg(long, value_enum, default_value_t = Architecture::Dense)]
        architecture: Architecture,
        #[arg(long)] model: PathBuf,
        #[arg(short, long, default_value = "training/mnist.json")] config: PathBuf,
    },
    Describe {
        #[arg(long, value_enum, default_value_t = Architecture::Dense)] architecture: Architecture,
        model: PathBuf,
    },
    Inspect { model: PathBuf },
    #[cfg(feature = "gpu")]
    GpuInfo,
}

#[derive(Clone, Copy, ValueEnum)]
enum Optimizer { Sgd, Momentum, Adam, Adamw }

#[derive(Clone, Copy, ValueEnum)]
enum LayerMethod { Passthrough, Gaussian, Copy, Values }

#[derive(Clone, Copy, ValueEnum)]
enum NeuronMethod { Passthrough, Gaussian }

#[derive(Clone, Copy, ValueEnum)]
enum ImportTarget { Token, Position, OutputHead }

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Architecture { Dense, Cnn, MemoryBank, Transformer }

impl Architecture {
    fn label(self) -> &'static str { match self { Architecture::Dense => "dense", Architecture::Cnn => "CNN", Architecture::MemoryBank => "memory bank", Architecture::Transformer => "transformer" } }
}

fn main() -> Result<(), Box<dyn Error>> {
    match Cli::parse_from(normalized_arguments()).command {
        Commands::New { architecture, layers, hidden, slots, blocks, vocab, heads, ff_hidden, max_sequence, output } => {
            let built = match architecture {
                Architecture::Dense => {
                    if hidden.is_some() || blocks.is_some() { return Err("--hidden and --blocks apply to cnn, memory-bank, and transformer models; dense models are shaped by --layers".into()); }
                    model::Model::dense(parse_layers(&layers).map_err(io::Error::other)?, slots.unwrap_or(0))
                }
                Architecture::Cnn => model::Model::mnist_convolutional(hidden.unwrap_or(64), slots.unwrap_or(0), blocks.unwrap_or(1)),
                Architecture::MemoryBank => model::Model::mnist_memory_bank(hidden.unwrap_or(128), slots.unwrap_or(32), blocks.unwrap_or(3)),
                Architecture::Transformer => {
                    let d_model = hidden.unwrap_or(128);
                    model::Model::language_model(vocab, d_model, heads, ff_hidden.unwrap_or(d_model * 4), blocks.unwrap_or(4), max_sequence)
                }
            }.map_err(io::Error::other)?;
            save_to(&built, &output)?;
            println!("Created {} model at {}", architecture.label(), output.display());
        }
        Commands::TrainText { model, output, text, vocab, merges, sequence, epochs, learning_rate, max_sequences, log_every, validation_fraction, growth, growth_max_units, growth_trigger, growth_patience, shrink_trigger, shrink_patience, growth_new_layer_ratio, growth_max_blocks, growth_max_ff, growth_interval, cuda } => {
            enable_cuda(cuda)?;
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            let tokenizer = tokenizer::Tokenizer::load(&vocab, &merges).map_err(io::Error::other)?;
            let corpus = fs::read_to_string(&text)?;
            let ids = tokenizer.encode(&corpus).map_err(io::Error::other)?;
            if ids.len() < sequence + 1 { return Err(format!("corpus has {} tokens, need at least {}", ids.len(), sequence + 1).into()); }
            let mut windows: Vec<(Vec<u32>, Vec<u32>)> = ids.windows(sequence + 1).step_by(sequence).map(|window| (window[..sequence].to_vec(), window[1..].to_vec())).collect();
            windows.shuffle(&mut rand::rng());
            if let Some(limit) = max_sequences { windows.truncate(limit); }
            // Held out before any training so growth strategies are compared on unseen text; a
            // bigger grown model would otherwise win simply by memorising more of the corpus.
            let held_out = ((windows.len() as f32 * validation_fraction).round() as usize).clamp(1, windows.len().saturating_sub(1));
            let validation = windows.split_off(windows.len() - held_out);
            println!("Training on {} tokens: {} training sequences, {} held-out, {sequence} tokens each", ids.len(), windows.len(), validation.len());
            let mut controller = growth.map(|strategy| growth::GrowthController::new(strategy, growth_max_units, growth_trigger, growth_patience, shrink_trigger, shrink_patience, growth_new_layer_ratio, growth_max_blocks, growth_max_ff, growth_interval));
            let interrupted = Arc::new(AtomicBool::new(false));
            let signal_flag = Arc::clone(&interrupted);
            ctrlc::set_handler(move || { signal_flag.store(true, Ordering::Relaxed); }).map_err(io::Error::other)?;
            let started = std::time::Instant::now();
            let mut steps = 0usize;
            'training: for epoch in 1..=epochs {
                // Chunked so a long epoch reports progress and can grow as it goes.
                for chunk in windows.chunks(log_every) {
                    let loss = network.train_language_model(chunk, 1, learning_rate, learn_functions::LearningFunction::AdamW { weight_decay: 0.01 }, &interrupted, |_, _| {}).map_err(io::Error::other)?;
                    steps += chunk.len();
                    let (validation_loss, validation_accuracy) = evaluate_sequences(&network, &validation).map_err(io::Error::other)?;
                    let elapsed = started.elapsed().as_secs_f32();
                    let parameters: usize = network.parameter_total();
                    println!("epoch {epoch} step {steps}/{}: train {loss:.4} | val {validation_loss:.4} (ppl {:.1}, acc {:.1}%) | {parameters} params | {:.0} tok/s | {:.1}m", windows.len() * epochs, validation_loss.exp(), validation_accuracy * 100.0, (steps * sequence) as f32 / elapsed, elapsed / 60.0);
                    if let Some(controller) = controller.as_mut() {
                        if let Some(change) = controller.observe(&mut network, validation_accuracy, validation_loss).map_err(io::Error::other)? {
                            println!("  growth: {change} -> {} params", network.parameter_total());
                        }
                    }
                    if interrupted.load(Ordering::Relaxed) { break 'training; }
                }
            }
            let elapsed = started.elapsed().as_secs_f32();
            let (validation_loss, validation_accuracy) = evaluate_sequences(&network, &validation).map_err(io::Error::other)?;
            println!("Trained {steps} sequence steps in {elapsed:.1}s");
            println!("FINAL: val loss {validation_loss:.4} | perplexity {:.2} | accuracy {:.2}% | {} parameters", validation_loss.exp(), validation_accuracy * 100.0, network.parameter_total());
            let was_interrupted = interrupted.load(Ordering::Relaxed);
            let output = output.unwrap_or(model);
            if was_interrupted && !confirm_save(&output)? { println!("Discarded interrupted training changes."); return Ok(()); }
            save_to(&network, &output)?;
            println!("Saved language model to {}", output.display());
        }
        Commands::Generate { model, prompt, tokens, vocab, merges, temperature, top_k, cuda } => {
            enable_cuda(cuda)?;
            let network = model::load_model(&model).map_err(io::Error::other)?;
            let tokenizer = tokenizer::Tokenizer::load(&vocab, &merges).map_err(io::Error::other)?;
            let ids = tokenizer.encode(&prompt).map_err(io::Error::other)?;
            if ids.is_empty() { return Err("prompt is empty".into()); }
            let started = std::time::Instant::now();
            let generated = network.generate_with(&ids, tokens, network.max_sequence(), temperature, top_k).map_err(io::Error::other)?;
            let elapsed = started.elapsed().as_secs_f32();
            println!("{}", tokenizer.decode(&generated).map_err(io::Error::other)?);
            println!("[{tokens} tokens in {elapsed:.1}s, {:.2} tokens/s]", tokens as f32 / elapsed.max(1e-6));
        }
        Commands::ImportEmbeddings { model, output, safetensors: source, tensor, target } => {
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            let (shape, values) = safetensors::read_tensor(&source, &tensor).map_err(io::Error::other)?;
            if shape.len() != 2 { return Err(format!("embedding tensor must be two-dimensional, got {shape:?}").into()); }
            let target = match target { ImportTarget::Token => model::EmbeddingTarget::Token, ImportTarget::Position => model::EmbeddingTarget::Position, ImportTarget::OutputHead => model::EmbeddingTarget::OutputHead };
            let copied = network.import_embeddings(target, shape[0], shape[1], &values).map_err(io::Error::other)?;
            let output = output.unwrap_or(model);
            save_to(&network, &output)?;
            println!("Imported {copied} rows of width {} from {}: {}", shape[1], tensor, output.display());
        }
        Commands::Train { architecture, config, output, model, layers, learning_rate, optimizer, momentum, cuda, epochs, batch_size, max_samples } => {
            let rule = match optimizer {
                Optimizer::Sgd => learn_functions::LearningFunction::Sgd, Optimizer::Momentum => learn_functions::LearningFunction::Momentum { momentum }, Optimizer::Adam => learn_functions::LearningFunction::Adam, Optimizer::Adamw => learn_functions::LearningFunction::AdamW { weight_decay: 0.01 },
            };
            if learning_rate <= 0.0 { return Err("learning rate must be greater than zero".into()); }
            let network = match model {
                Some(path) => model::load_model(&path).map_err(io::Error::other)?,
                None => match architecture {
                    Architecture::Dense => model::Model::dense(parse_layers(&layers).map_err(io::Error::other)?, 0),
                    Architecture::Cnn => model::Model::mnist_convolutional(64, 0, 1),
                    Architecture::MemoryBank => model::Model::mnist_memory_bank(128, 32, 3),
                    Architecture::Transformer => return Err("transformer training needs an existing model; create one with `new --architecture transformer`".into()),
                }.map_err(io::Error::other)?,
            };
            let interrupted = Arc::new(AtomicBool::new(false));
            let signal_flag = Arc::clone(&interrupted);
            ctrlc::set_handler(move || { signal_flag.store(true, Ordering::Relaxed); }).map_err(io::Error::other)?;
            let overrides = training::Overrides { epochs, batch_size, max_samples };
            let outcome = training::train_from_json(network, config.to_str().ok_or("config path is not UTF-8")?, learning_rate, rule, &interrupted, cuda, overrides).map_err(io::Error::other)?;
            let was_interrupted = outcome.was_interrupted();
            if was_interrupted && !confirm_save(&output)? { println!("Discarded interrupted training changes."); return Ok(()); }
            save_to(&outcome.into_model(), &output)?;
            println!("Saved {}{} model to {}{}", if was_interrupted { "interrupted " } else { "trained " }, architecture.label(), output.display(), if cuda { " using CUDA" } else { "" });
        }
        Commands::AddLayer { model, output, layer_size, insert_at, method, copy_from, values_file } => {
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            let initialization = match method {
                LayerMethod::Passthrough => model::LayerInit::Passthrough,
                LayerMethod::Gaussian => model::LayerInit::Gaussian,
                LayerMethod::Copy => model::LayerInit::Copy { source_layer: copy_from.ok_or("--copy-from is required with --method copy")? },
                LayerMethod::Values => {
                    let path = values_file.ok_or("--values-file is required with --method values")?;
                    let text = fs::read_to_string(path)?;
                    model::LayerInit::Values(serde_json::from_str(&text).map_err(io::Error::other)?)
                }
            };
            network.insert_layer(insert_at, layer_size, initialization).map_err(io::Error::other)?;
            let output = output.unwrap_or(model);
            save_to(&network, &output)?;
            println!("Inserted layer of size {layer_size} after layer {insert_at}: {}", output.display());
        }
        Commands::AddNeurons { model, output, layer, insert_at, count, method } => {
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            network.add_neurons(layer, insert_at, count, matches!(method, NeuronMethod::Gaussian)).map_err(io::Error::other)?;
            let output = output.unwrap_or(model);
            save_to(&network, &output)?;
            println!("Added {count} neuron(s) at index {insert_at} in layer {layer}: {}", output.display());
        }
        Commands::RemoveNeurons { model, output, layer, indexes } => {
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            let indexes = parse_neuron_indexes(&indexes).map_err(io::Error::other)?;
            network.remove_neurons(layer, &indexes).map_err(io::Error::other)?;
            let output = output.unwrap_or(model);
            save_to(&network, &output)?;
            println!("Removed neuron(s) from layer {layer}: {}", output.display());
        }
        Commands::RemoveLayer { model, output, layer } => {
            let mut network = model::load_model(&model).map_err(io::Error::other)?;
            network.remove_layer(layer).map_err(io::Error::other)?;
            let output = output.unwrap_or(model);
            save_to(&network, &output)?;
            println!("Removed layer {layer}: {}", output.display());
        }
        Commands::Evaluate { architecture: _, model, config } => {
            let network = model::load_model(&model).map_err(io::Error::other)?;
            let accuracy = training::evaluate_from_json(&network, config.to_str().ok_or("config path is not UTF-8")?).map_err(io::Error::other)?;
            println!("Test accuracy: {:.2}%", accuracy * 100.0);
        }
        Commands::Describe { architecture: _, model } => {
            println!("Model: {}", model.display());
            println!("{}", model::load_model(&model).map_err(io::Error::other)?.describe());
        }
        Commands::Inspect { model } => {
            model::load_model(&model).map_err(io::Error::other)?;
            println!("Valid neuralnet GGUF model: {}", model.display());
        }
        #[cfg(feature = "gpu")]
        Commands::GpuInfo => println!("{}", gpu::probe().map_err(io::Error::other)?),
    }
    Ok(())
}

fn normalized_arguments() -> Vec<OsString> {
    normalize_arguments(std::env::args_os().collect())
}

fn normalize_arguments(arguments: Vec<OsString>) -> Vec<OsString> {
    arguments.into_iter().map(|argument| if argument == "--train" { OsString::from("train") } else { argument }).collect()
}

/// Mean next-token loss and accuracy over held-out sequences.
fn evaluate_sequences(network: &model::Model, sequences: &[(Vec<u32>, Vec<u32>)]) -> Result<(f32, f32), String> {
    if sequences.is_empty() { return Err("no held-out sequences".into()); }
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut total = 0usize;
    for (tokens, targets) in sequences {
        let (sequence_loss, predictions) = network.language_model_evaluate(tokens, targets)?;
        loss += sequence_loss;
        correct += predictions;
        total += targets.len();
    }
    Ok((loss / sequences.len() as f32, correct as f32 / total as f32))
}

fn enable_cuda(requested: bool) -> Result<(), Box<dyn Error>> {
    if !requested { return Ok(()); }
    #[cfg(feature = "gpu")]
    {
        gpu::set_enabled(true);
        Ok(())
    }
    #[cfg(not(feature = "gpu"))]
    Err("--cuda requires building with `--features gpu`".into())
}

fn save_to(network: &model::Model, path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    model::save_model(network, path).map_err(io::Error::other)?;
    Ok(())
}

fn parse_layers(value: &str) -> Result<Vec<usize>, String> {
    let layers: Result<Vec<_>, _> = value.split(',').map(str::parse::<usize>).collect();
    let layers = layers.map_err(|_| "layers must be comma-separated positive integers".to_owned())?;
    if layers.len() < 2 || layers.contains(&0) { return Err("layers requires at least two positive sizes, such as 784,128,10".to_owned()); }
    Ok(layers)
}

fn parse_neuron_indexes(value: &str) -> Result<Vec<usize>, String> {
    let mut indexes = Vec::new();
    let mut items = Vec::new();
    let mut start = 0;
    let mut in_range = false;
    for (index, character) in value.char_indices() {
        match character {
            '[' if !in_range => in_range = true,
            ']' if in_range => in_range = false,
            ',' if !in_range => { items.push(&value[start..index]); start = index + 1; }
            _ => {}
        }
    }
    if in_range { return Err("ranges must end with ']'".to_owned()); }
    items.push(&value[start..]);
    for item in items {
        let item = item.trim();
        if let Some(range) = item.strip_prefix('[').and_then(|item| item.strip_suffix(']')) {
            let (start, end) = range.split_once(',').ok_or("ranges must use [start,end], for example [0,63]")?;
            let start = start.trim().parse::<usize>().map_err(|_| "range start must be a non-negative integer")?;
            let end = end.trim().parse::<usize>().map_err(|_| "range end must be a non-negative integer")?;
            if start > end { return Err("range start must not be greater than range end".to_owned()); }
            indexes.extend(start..=end);
        } else {
            indexes.push(item.parse::<usize>().map_err(|_| "indexes must be integers or inclusive [start,end] ranges")?);
        }
    }
    if indexes.is_empty() { return Err("provide at least one neuron index".to_owned()); }
    Ok(indexes)
}

fn confirm_save(path: &PathBuf) -> Result<bool, io::Error> {
    loop {
        print!("Training interrupted. Save checkpoint to {}? [s]ave/[d]iscard: ", path.display());
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        match answer.trim().to_ascii_lowercase().as_str() {
            "s" | "save" => return Ok(true),
            "d" | "discard" => return Ok(false),
            _ => eprintln!("Enter 's' to save or 'd' to discard."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_arguments, parse_neuron_indexes};
    use std::ffi::OsString;

    #[test]
    fn expands_neuron_indexes_and_inclusive_ranges() {
        assert_eq!(parse_neuron_indexes("[0,63],71,80").unwrap(), (0..=63).chain([71, 80]).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_invalid_neuron_ranges() {
        assert!(parse_neuron_indexes("[63,0]").is_err());
        assert!(parse_neuron_indexes("[0]").is_err());
        assert!(parse_neuron_indexes("not-an-index").is_err());
    }

    #[test]
    fn normalizes_train_flag_to_subcommand() {
        let arguments = normalize_arguments(vec![OsString::from("neuralnet"), OsString::from("--train"), OsString::from("--cuda")]);
        assert_eq!(arguments, vec![OsString::from("neuralnet"), OsString::from("train"), OsString::from("--cuda")]);
    }
}

