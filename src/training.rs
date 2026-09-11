use crate::{
    learn_functions::LearningFunction,
    model::Model,
    data::{training_sets, validation_sets, TrainingData},
};
use rand::seq::SliceRandom;
use std::{
    fs,
    sync::atomic::{AtomicBool, Ordering},
};

pub enum TrainingOutcome {
    Complete(Model),
    Interrupted(Model),
}

impl TrainingOutcome {
    pub fn into_model(self) -> Model {
        match self {
            TrainingOutcome::Complete(model) | TrainingOutcome::Interrupted(model) => model,
        }
    }
    pub fn was_interrupted(&self) -> bool { matches!(self, TrainingOutcome::Interrupted(_)) }
}

pub struct Overrides {
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub max_samples: Option<usize>,
}

fn read_config(training_file: &str) -> Result<TrainingData, String> {
    serde_json::from_str(&fs::read_to_string(training_file).map_err(|error| error.to_string())?).map_err(|error| format!("invalid training JSON: {error}"))
}

/// Trains against a JSON dataset config, tracking the best validation checkpoint.
pub fn train_from_json(mut model: Model, training_file: &str, learning_rate: f32, rule: LearningFunction, interrupted: &AtomicBool, use_cuda: bool, overrides: Overrides) -> Result<TrainingOutcome, String> {
    let config = read_config(training_file)?;
    if config.target_accuracy.is_some_and(|target| !(0.0..=1.0).contains(&target)) { return Err("target_accuracy must be between 0.0 and 1.0".into()); }
    let use_cuda = use_cuda || config.use_cuda;
    let max_epochs = overrides.epochs.unwrap_or(config.max_epochs);
    let batch_size = overrides.batch_size.unwrap_or(config.batch_size);
    println!("Training {} for at most {} epoch(s) with batches of {}{}", config.name, max_epochs, batch_size, if use_cuda { " on CUDA" } else { "" });

    let (mut inputs, mut outputs) = training_sets(&config)?;
    if let Some(max) = overrides.max_samples.or(config.max_samples) { inputs.truncate(max); outputs.truncate(max); }
    let (validation_inputs, validation_outputs) = validation_sets(&config, &mut inputs, &mut outputs)?;
    if inputs.is_empty() || validation_inputs.is_empty() { return Err("training and validation splits must both contain samples".into()); }
    if config.validation.is_none() && validation_inputs.len() == inputs.len() {
        println!("WARNING: validation_samples ({}) does not leave a held-out split for {} training samples, so reported accuracy is measured on the training data itself.", config.validation_samples, inputs.len());
    }
    println!("Using {} training samples and {} validation samples", inputs.len(), validation_inputs.len());
    println!("Initial validation accuracy {:.2}%", model.accuracy(&validation_inputs, &validation_outputs)? * 100.0);

    let mut best_model = model.clone();
    let mut best_accuracy = 0.0;
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    let mut rng = rand::rng();
    for epoch in 1..=max_epochs {
        if interrupted.load(Ordering::Relaxed) { return Ok(TrainingOutcome::Interrupted(best_model)); }
        order.shuffle(&mut rng);
        if model.train_epoch(&inputs, &outputs, &order, batch_size, learning_rate, rule, use_cuda, interrupted)? {
            return Ok(TrainingOutcome::Interrupted(best_model));
        }
        let accuracy = model.accuracy(&validation_inputs, &validation_outputs)?;
        println!("Epoch {epoch}: validation accuracy {:.2}%", accuracy * 100.0);
        if accuracy > best_accuracy + 0.0001 {
            best_accuracy = accuracy;
            best_model = model.clone();
        }
        if config.target_accuracy.is_some_and(|target| accuracy >= target) {
            println!("Reached target accuracy at epoch {epoch}");
            return Ok(TrainingOutcome::Complete(best_model));
        }
    }
    Ok(TrainingOutcome::Complete(best_model))
}

/// Accuracy on the config's held-out `test` source.
pub fn evaluate_from_json(model: &Model, training_file: &str) -> Result<f32, String> {
    let (inputs, outputs) = crate::data::test_pairs_from_json(training_file)?;
    model.accuracy(&inputs, &outputs)
}
