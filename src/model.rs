use crate::{gguf, learn_functions::LearningFunction};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{fs, path::{Path, PathBuf}, sync::atomic::{AtomicBool, Ordering}};
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchTensor {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    },
    Relu,
    MaxPool2d { kernel_size: usize },
    Flatten,
    Dense { inputs: usize, outputs: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Appends a learned projection of the current activation to a bank that every later layer reads.
    MemoryBank { inputs: usize, slots: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Maps token ids in a `1 x 1 x sequence` tensor to a `1 x sequence x d_model` tensor.
    Embedding { vocab: usize, d_model: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Adds learned position vectors to a `1 x sequence x d_model` tensor.
    PositionalEmbedding { max_sequence: usize, d_model: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Pre-norm block: causal multi-head self-attention and a GELU feed-forward network.
    TransformerBlock { d_model: usize, heads: usize, ff_hidden: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Row-wise layer normalisation over the last dimension.
    LayerNorm { size: usize, weights: Vec<f32>, biases: Vec<f32> },
    /// Applies one dense projection independently to every row of a sequence.
    TimeDistributedDense { inputs: usize, outputs: usize, weights: Vec<f32>, biases: Vec<f32> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub layers: Vec<Layer>,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    optimizer: Option<ModelOptimizer>,
}

struct ForwardCache {
    activations: Vec<Tensor>,
    /// Bank length before each layer, plus the final length at the end.
    bank_lengths: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ModelOptimizer {
    function: LearningFunction,
    step: u32,
    first: Vec<Vec<f32>>,
    second: Vec<Vec<f32>>,
}

impl ModelOptimizer {
    fn matches(&self, model: &Model) -> bool {
        let sizes = model.parameter_sizes();
        self.first.len() == sizes.len() && self.second.len() == sizes.len()
            && self.first.iter().zip(&sizes).all(|(buffer, size)| buffer.len() == *size)
            && self.second.iter().zip(&sizes).all(|(buffer, size)| buffer.len() == *size)
    }
}

/// A gradient that may live on the host or, when CUDA is active, stay in device memory from the
/// backward pass all the way through the optimizer step.
pub(crate) enum GradientData {
    Host(Vec<f32>),
    #[cfg(feature = "gpu")]
    Device(crate::gpu::DeviceVector),
}

impl From<Vec<f32>> for GradientData {
    fn from(values: Vec<f32>) -> Self { GradientData::Host(values) }
}

impl GradientData {
    pub(crate) fn len(&self) -> usize {
        match self {
            GradientData::Host(values) => values.len(),
            #[cfg(feature = "gpu")]
            GradientData::Device(buffer) => buffer.len(),
        }
    }

    /// Host copy of the gradient, downloading it if it currently lives on the device.
    pub(crate) fn to_host(&self) -> Vec<f32> {
        match self {
            GradientData::Host(values) => values.clone(),
            #[cfg(feature = "gpu")]
            GradientData::Device(buffer) => buffer.to_host().unwrap_or_default(),
        }
    }

    fn add_assign(&mut self, other: &GradientData) -> Result<(), String> {
        match (self, other) {
            (GradientData::Host(total), GradientData::Host(sample)) => {
                for (total, sample) in total.iter_mut().zip(sample) { *total += sample; }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            (GradientData::Device(total), GradientData::Device(sample)) => crate::gpu::device_add(total, sample),
            #[cfg(feature = "gpu")]
            (total, other) => {
                // Mixed representations: fall back to host arithmetic rather than guessing.
                let mut values = total.to_host();
                for (value, sample) in values.iter_mut().zip(other.to_host()) { *value += sample; }
                *total = GradientData::Host(values);
                Ok(())
            }
        }
    }

    fn scale(&mut self, factor: f32) -> Result<(), String> {
        match self {
            GradientData::Host(values) => { for value in values { *value *= factor; } Ok(()) }
            #[cfg(feature = "gpu")]
            GradientData::Device(buffer) => crate::gpu::device_scale(buffer, factor),
        }
    }
}

pub(crate) struct LayerGradient { pub(crate) weights: GradientData, pub(crate) biases: GradientData }

impl Tensor {
    pub fn new(channels: usize, height: usize, width: usize, values: Vec<f32>) -> Result<Self, String> {
        if values.len() != channels * height * width { return Err("tensor values do not match its shape".into()); }
        Ok(Self { channels, height, width, values })
    }
}

impl BatchTensor {
    pub fn new(batch: usize, channels: usize, height: usize, width: usize, values: Vec<f32>) -> Result<Self, String> {
        if values.len() != batch * channels * height * width { return Err("batch tensor values do not match its shape".into()); }
        Ok(Self { batch, channels, height, width, values })
    }
}

pub(crate) fn batch_conv2d(input: &BatchTensor, out_channels: usize, kernel: usize, weights: &[f32], biases: &[f32]) -> Result<BatchTensor, String> {
    let mut values = Vec::new();
    for batch in 0..input.batch {
        let start = batch * input.channels * input.height * input.width;
        values.extend(conv2d(&Tensor::new(input.channels, input.height, input.width, input.values[start..start + input.channels * input.height * input.width].to_vec())?, input.channels, out_channels, kernel, weights, biases)?.values);
    }
    BatchTensor::new(input.batch, out_channels, input.height - kernel + 1, input.width - kernel + 1, values)
}

pub(crate) fn batch_max_pool2d(input: &BatchTensor, kernel: usize) -> Result<BatchTensor, String> {
    let mut values = Vec::new();
    for batch in 0..input.batch {
        let start = batch * input.channels * input.height * input.width;
        values.extend(max_pool2d(&Tensor::new(input.channels, input.height, input.width, input.values[start..start + input.channels * input.height * input.width].to_vec())?, kernel)?.values);
    }
    BatchTensor::new(input.batch, input.channels, input.height / kernel, input.width / kernel, values)
}

pub(crate) fn batch_relu_backward(input: &BatchTensor, gradient: &BatchTensor) -> Result<BatchTensor, String> {
    if input != gradient && (input.batch != gradient.batch || input.channels != gradient.channels || input.height != gradient.height || input.width != gradient.width) { return Err("batched ReLU gradient shape does not match input".into()); }
    BatchTensor::new(input.batch, input.channels, input.height, input.width, input.values.iter().zip(&gradient.values).map(|(value, gradient)| if *value > 0.0 { *gradient } else { 0.0 }).collect())
}

pub(crate) fn batch_max_pool2d_backward(input: &BatchTensor, kernel: usize, gradient: &BatchTensor) -> Result<BatchTensor, String> {
    let pooled = batch_max_pool2d(input, kernel)?;
    if pooled.batch != gradient.batch || pooled.channels != gradient.channels || pooled.height != gradient.height || pooled.width != gradient.width { return Err("batched max-pool gradient shape does not match output".into()); }
    let mut values = vec![0.0; input.values.len()];
    let input_size = input.channels * input.height * input.width;
    let output_size = pooled.channels * pooled.height * pooled.width;
    for batch in 0..input.batch { for channel in 0..pooled.channels { for row in 0..pooled.height { for column in 0..pooled.width {
        let mut maximum = f32::NEG_INFINITY; let mut maximum_index = 0;
        for ky in 0..kernel { for kx in 0..kernel { let index = batch * input_size + channel * input.height * input.width + (row * kernel + ky) * input.width + column * kernel + kx; if input.values[index] > maximum { maximum = input.values[index]; maximum_index = index; } } }
        values[maximum_index] = gradient.values[batch * output_size + channel * pooled.height * pooled.width + row * pooled.width + column];
    } } } }
    BatchTensor::new(input.batch, input.channels, input.height, input.width, values)
}

/// Every model construction, drop, and parameter write must bump the device cache epoch, or a
/// cached GPU buffer could outlive the host data it mirrors.
#[inline]
fn invalidate_device_cache() {
    #[cfg(feature = "gpu")]
    crate::gpu::invalidate_device_cache();
}

impl Drop for Model {
    fn drop(&mut self) { invalidate_device_cache(); }
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self { Self::with_input_shape(1, 28, 28, layers) }

    pub fn with_input_shape(channels: usize, height: usize, width: usize, layers: Vec<Layer>) -> Self {
        invalidate_device_cache();
        Self { layers, input_channels: channels, input_height: height, input_width: width, optimizer: None }
    }

    /// Fully connected ReLU stack matching the historical dense engine, optionally with memory banks.
    pub fn dense(layer_sizes: Vec<usize>, slots: usize) -> Result<Self, String> {
        if layer_sizes.len() < 2 || layer_sizes.contains(&0) { return Err("a network needs at least two non-empty layers".into()); }
        let mut rng = rand::rng();
        let mut layers = vec![Layer::Flatten];
        let mut bank_total = 0;
        for (index, pair) in layer_sizes.windows(2).enumerate() {
            let hidden = index + 2 < layer_sizes.len();
            let inputs = pair[0] + if index == 0 { 0 } else { bank_total };
            layers.push(Layer::Dense { inputs, outputs: pair[1], weights: uniform_weights(inputs * pair[1], inputs, &mut rng), biases: vec![0.0; pair[1]] });
            if hidden {
                layers.push(Layer::Relu);
                if slots > 0 {
                    layers.push(Layer::MemoryBank { inputs: pair[1], slots, weights: uniform_weights(pair[1] * slots, pair[1], &mut rng), biases: vec![0.0; slots] });
                    bank_total += slots;
                }
            }
        }
        Ok(Self::with_input_shape(1, 1, layer_sizes[0], layers))
    }

    /// Dense layer widths, for models whose graph is a plain fully connected stack.
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        for layer in &self.layers {
            if let Layer::Dense { inputs, outputs, .. } = layer {
                if sizes.is_empty() { sizes.push(*inputs); }
                sizes.push(*outputs);
            }
        }
        sizes
    }

    pub fn mnist() -> Self { Self::mnist_convolutional(64, 0, 1).expect("default convolutional geometry is valid") }

    /// Convolutional stem feeding a memory bank head. `slots` of 0 gives a plain dense head.
    pub fn mnist_convolutional(hidden: usize, slots: usize, blocks: usize) -> Result<Self, String> {
        if hidden == 0 || blocks == 0 { return Err("models need positive hidden and block counts".into()); }
        let mut rng = rand::rng();
        let conv1 = Layer::Conv2d { in_channels: 1, out_channels: 8, kernel_size: 3, weights: gaussian_weights(8 * 3 * 3, 9, &mut rng), biases: vec![0.0; 8] };
        let conv2 = Layer::Conv2d { in_channels: 8, out_channels: 16, kernel_size: 3, weights: gaussian_weights(16 * 8 * 3 * 3, 72, &mut rng), biases: vec![0.0; 16] };
        let mut layers = vec![conv1, Layer::Relu, Layer::MaxPool2d { kernel_size: 2 }, conv2, Layer::Relu, Layer::MaxPool2d { kernel_size: 2 }, Layer::Flatten];
        layers.extend(memory_bank_head(16 * 5 * 5, hidden, slots, blocks, 10, &mut rng));
        Ok(Self::with_input_shape(1, 28, 28, layers))
    }

    /// Fully connected stack where every block reads the previous block plus the whole accumulated bank.
    pub fn mnist_memory_bank(hidden: usize, slots: usize, blocks: usize) -> Result<Self, String> {
        if hidden == 0 || slots == 0 || blocks == 0 { return Err("memory bank models need positive hidden, slot, and block counts".into()); }
        let mut rng = rand::rng();
        let mut layers = vec![Layer::Flatten];
        layers.extend(memory_bank_head(28 * 28, hidden, slots, blocks, 10, &mut rng));
        Ok(Self::with_input_shape(1, 28, 28, layers))
    }

    pub fn describe(&self) -> String {
        let parameters: usize = self.parameter_sizes().iter().sum();
        let mut details = format!("Input: {}x{}x{}\nLayers: {}\nTotal parameters: {}", self.input_channels, self.input_height, self.input_width, self.layers.len(), parameters);
        for (index, layer) in self.layers.iter().enumerate() {
            let line = match layer {
                Layer::Conv2d { in_channels, out_channels, kernel_size, weights, biases } => format!("  {index}: Conv2d {in_channels}->{out_channels}, {kernel_size}x{kernel_size}, parameters: {}", weights.len() + biases.len()),
                Layer::Relu => format!("  {index}: ReLU"),
                Layer::MaxPool2d { kernel_size } => format!("  {index}: MaxPool2d {kernel_size}x{kernel_size}"),
                Layer::Flatten => format!("  {index}: Flatten"),
                Layer::Dense { inputs, outputs, weights, biases } => format!("  {index}: Dense {inputs}->{outputs}, parameters: {}", weights.len() + biases.len()),
                Layer::MemoryBank { inputs, slots, weights, biases } => format!("  {index}: MemoryBank {inputs}->{slots} slots, parameters: {}", weights.len() + biases.len()),
                Layer::Embedding { vocab, d_model, weights, .. } => format!("  {index}: Embedding {vocab} tokens x {d_model}, parameters: {}", weights.len()),
                Layer::PositionalEmbedding { max_sequence, d_model, weights, .. } => format!("  {index}: PositionalEmbedding {max_sequence} x {d_model}, parameters: {}", weights.len()),
                Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases } => format!("  {index}: TransformerBlock d_model {d_model}, heads {heads}, ff {ff_hidden}, parameters: {}", weights.len() + biases.len()),
                Layer::LayerNorm { size, weights, biases } => format!("  {index}: LayerNorm {size}, parameters: {}", weights.len() + biases.len()),
                Layer::TimeDistributedDense { inputs, outputs, weights, biases } => format!("  {index}: TimeDistributedDense {inputs}->{outputs}, parameters: {}", weights.len() + biases.len()),
            };
            details.push('\n'); details.push_str(&line);
        }
        details
    }

    pub fn forward(&self, mut tensor: Tensor) -> Result<Vec<f32>, String> {
        let mut bank: Vec<f32> = Vec::new();
        for layer in &self.layers {
            tensor = match layer {
                Layer::Conv2d { in_channels, out_channels, kernel_size, weights, biases } => conv2d(&tensor, *in_channels, *out_channels, *kernel_size, weights, biases)?,
                Layer::Relu => Tensor { values: tensor.values.iter().map(|value| value.max(0.0)).collect(), ..tensor },
                Layer::MaxPool2d { kernel_size } => max_pool2d(&tensor, *kernel_size)?,
                Layer::Flatten => Tensor { channels: 1, height: 1, width: tensor.values.len(), values: tensor.values },
                Layer::Dense { inputs, outputs, weights, biases } => dense(&tensor, *inputs, *outputs, weights, biases)?,
                Layer::MemoryBank { inputs, slots, weights, biases } => {
                    bank.extend(memory_bank_write(&tensor, *inputs, *slots, weights, biases)?);
                    let mut values = tensor.values;
                    values.extend_from_slice(&bank);
                    Tensor { channels: 1, height: 1, width: values.len(), values }
                }
                _ => sequence_forward(layer, &tensor)?,
            };
        }
        Ok(softmax(tensor.values))
    }

    fn forward_cached(&self, input: Tensor) -> Result<ForwardCache, String> {
        self.forward_cached_through(input, self.layers.len())
    }

    fn forward_cached_through(&self, input: Tensor, layer_count: usize) -> Result<ForwardCache, String> {
        let mut activations = vec![input];
        let mut bank: Vec<f32> = Vec::new();
        let mut bank_lengths = vec![0];
        for layer in self.layers.iter().take(layer_count) {
            let tensor = activations.last().unwrap();
            let next = match layer {
                Layer::Conv2d { in_channels, out_channels, kernel_size, weights, biases } => conv2d(tensor, *in_channels, *out_channels, *kernel_size, weights, biases)?,
                Layer::Relu => Tensor { values: tensor.values.iter().map(|value| value.max(0.0)).collect(), ..tensor.clone() },
                Layer::MaxPool2d { kernel_size } => max_pool2d(tensor, *kernel_size)?,
                Layer::Flatten => Tensor { channels: 1, height: 1, width: tensor.values.len(), values: tensor.values.clone() },
                Layer::Dense { inputs, outputs, weights, biases } => dense(tensor, *inputs, *outputs, weights, biases)?,
                Layer::MemoryBank { inputs, slots, weights, biases } => {
                    bank.extend(memory_bank_write(tensor, *inputs, *slots, weights, biases)?);
                    let mut values = tensor.values.clone();
                    values.extend_from_slice(&bank);
                    Tensor { channels: 1, height: 1, width: values.len(), values }
                }
                _ => sequence_forward(layer, tensor)?,
            };
            bank_lengths.push(bank.len());
            activations.push(next);
        }
        Ok(ForwardCache { activations, bank_lengths })
    }

    pub fn train(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], batch_size: usize, epochs: usize, learning_rate: f32, function: LearningFunction) -> Result<(), String> {
        let interrupted = AtomicBool::new(false);
        self.train_with_progress(inputs, targets, batch_size, epochs, learning_rate, function, false, &interrupted, |_, _| {}).map(|_| ())
    }

    fn sample_gradients(&self, input: &[f32], target: &[f32], use_cuda: bool) -> Result<Vec<Option<LayerGradient>>, String> {
        #[cfg(feature = "gpu")]
        if use_cuda { return self.cuda_backward(input, target); }
        #[cfg(not(feature = "gpu"))]
        if use_cuda { return Err("CUDA training requires building with `--features gpu`".into()); }
        self.backward(input, target)
    }

    /// Runs the optimizer update on the GPU, leaving the moments resident there.
    #[cfg(feature = "gpu")]
    fn apply_gradients_on_device(&mut self, gradients: &[Option<LayerGradient>], learning_rate: f32, optimizer: &mut ModelOptimizer) -> Result<(), String> {
        let decay = match optimizer.function { LearningFunction::AdamW { weight_decay } => weight_decay, _ => 0.0 };
        let mut parameter_index = 0;
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
            // Embedding rows are gathered from the device and blocks and the head run there
            // entirely, so those tables may stay device-resident between steps.
            // A block's packed weights are also cached as sub-slices, which are evicted on
            // update, so its host copy must stay current for them to re-upload correctly.
            let host_forward_is_device_resident = matches!(layer, Layer::Embedding { .. } | Layer::TimeDistributedDense { .. });
            let (Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. }, Some(gradient)) = (layer, gradient) else { continue };
            for (parameters, data) in [(&mut *weights, &gradient.weights), (&mut *biases, &gradient.biases)] {
                match data {
                    GradientData::Device(buffer) => crate::gpu::adamw_step_device(parameters, buffer, &mut optimizer.first[parameter_index], &mut optimizer.second[parameter_index], learning_rate, decay, optimizer.step)?,
                    GradientData::Host(values) => crate::gpu::adamw_step(parameters, values, &mut optimizer.first[parameter_index], &mut optimizer.second[parameter_index], learning_rate, decay, optimizer.step)?,
                }
                // Only layers whose forward pass runs entirely on the device may stay dirty; the
                // rest are read from host memory next step and must be brought back now.
                if !host_forward_is_device_resident { crate::gpu::flush_parameters(parameters)?; }
                parameter_index += 1;
            }
        }
        Ok(())
    }

    /// Copies device-resident parameters and optimizer moments back to the host.
    fn flush_device_state(&mut self, optimizer: &mut ModelOptimizer) {
        #[cfg(feature = "gpu")]
        {
            for layer in &mut self.layers {
                if let Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. } = layer {
                    let _ = crate::gpu::flush_parameters(weights);
                    let _ = crate::gpu::flush_parameters(biases);
                }
            }
            for (first, second) in optimizer.first.iter_mut().zip(optimizer.second.iter_mut()) {
                let _ = crate::gpu::flush_moments(first, second);
            }
        }
        #[cfg(not(feature = "gpu"))]
        let _ = optimizer;
    }

    fn take_optimizer(&mut self, function: LearningFunction) -> ModelOptimizer {
        let sizes = self.parameter_sizes();
        self.optimizer.take()
            .filter(|state| state.function == function && state.first.iter().map(Vec::len).eq(sizes.iter().copied()))
            .unwrap_or_else(|| {
                // Fresh host buffers could land on addresses that still map to old device moments.
                #[cfg(feature = "gpu")]
                crate::gpu::clear_moment_buffers();
                ModelOptimizer { function, step: 0, first: sizes.iter().map(|&size| vec![0.0; size]).collect(), second: sizes.iter().map(|&size| vec![0.0; size]).collect() }
            })
    }

    /// Dense connections for the batched CUDA kernel, which assumes ReLU hidden layers and a
    /// linear softmax output. Returns None whenever the graph deviates from that exact shape.
    #[cfg(feature = "gpu")]
    pub(crate) fn dense_cuda_chain(&self) -> Option<(Vec<usize>, Vec<crate::gpu::DenseLayer>)> {
        let mut body = self.layers.as_slice();
        if let [Layer::Flatten, rest @ ..] = body { body = rest; }
        let mut layers = Vec::new();
        loop {
            match body {
                [Layer::Dense { weights, biases, .. }] => {
                    layers.push(crate::gpu::DenseLayer { weights: weights.clone(), biases: biases.clone() });
                    break;
                }
                [Layer::Dense { weights, biases, .. }, Layer::Relu, rest @ ..] => {
                    layers.push(crate::gpu::DenseLayer { weights: weights.clone(), biases: biases.clone() });
                    body = rest;
                }
                _ => return None,
            }
        }
        Some((self.layer_sizes(), layers))
    }

    /// Runs one pass over `order`. Returns true when interrupted at a batch boundary.
    pub fn train_epoch(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], order: &[usize], batch_size: usize, learning_rate: f32, function: LearningFunction, use_cuda: bool, interrupted: &AtomicBool) -> Result<bool, String> {
        if inputs.is_empty() || inputs.len() != targets.len() || batch_size == 0 { return Err("inputs, targets, and batch size are invalid".into()); }
        let mut optimizer = self.take_optimizer(function);
        for batch in order.chunks(batch_size) {
            if interrupted.load(Ordering::Relaxed) { self.flush_device_state(&mut optimizer); self.optimizer = Some(optimizer); return Ok(true); }
            let mut gradients = self.zero_gradients();
            let mut batched = false;
            #[cfg(feature = "gpu")]
            if use_cuda {
                if let Some((sizes, dense)) = self.dense_cuda_chain() {
                    let batch_inputs: Vec<Vec<f32>> = batch.iter().map(|&index| inputs[index].clone()).collect();
                    let batch_targets: Vec<Vec<f32>> = batch.iter().map(|&index| targets[index].clone()).collect();
                    let gradient = crate::gpu::batch_gradient(&sizes, &dense, &batch_inputs, &batch_targets)?;
                    for (connection, slot) in gradients.iter_mut().flatten().enumerate() {
                        slot.weights = gradient.weights[connection].clone().into();
                        slot.biases = gradient.biases[connection].clone().into();
                    }
                    batched = true;
                }
            }
            if !batched {
                for &index in batch {
                    let sample = self.sample_gradients(&inputs[index], &targets[index], use_cuda)?;
                    add_gradients(&mut gradients, &sample);
                }
                scale_gradients(&mut gradients, 1.0 / batch.len() as f32);
            }
            optimizer.step += 1;
            self.apply_gradients(&gradients, learning_rate, &mut optimizer);
        }
        self.flush_device_state(&mut optimizer);
        self.optimizer = Some(optimizer);
        Ok(false)
    }

    pub fn train_with_progress(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], batch_size: usize, epochs: usize, learning_rate: f32, function: LearningFunction, use_cuda: bool, interrupted: &AtomicBool, mut progress: impl FnMut(usize, f32)) -> Result<bool, String> {
        if inputs.is_empty() || inputs.len() != targets.len() || batch_size == 0 { return Err("inputs, targets, and batch size are invalid".into()); }
        let mut order: Vec<_> = (0..inputs.len()).collect();
        let mut rng = rand::rng();
        for epoch in 1..=epochs {
            if interrupted.load(Ordering::Relaxed) { return Ok(true); }
            order.shuffle(&mut rng);
            if self.train_epoch(inputs, targets, &order, batch_size, learning_rate, function, use_cuda, interrupted)? { return Ok(true); }
            progress(epoch, self.accuracy(inputs, targets)?);
        }
        Ok(false)
    }

    /// Next-token cross-entropy over a sequence, plus gradients for every parameter.
    ///
    /// `tokens` and `targets` have equal length; position `t` predicts `targets[t]`.
    pub fn language_model_step(&self, tokens: &[u32], targets: &[u32]) -> Result<(f32, Vec<Option<LayerGradient>>), String> {
        if tokens.is_empty() || tokens.len() != targets.len() { return Err("token and target sequences must be equal and non-empty".into()); }
        let input = Tensor::new(1, 1, tokens.len(), tokens.iter().map(|token| *token as f32).collect())?;

        #[cfg(feature = "gpu")]
        if crate::gpu::enabled() {
            let output_index = self.layers.len().saturating_sub(1);
            if let Some(Layer::TimeDistributedDense { inputs, outputs, weights, biases }) = self.layers.last() {
                let cache = self.forward_cached_through(input.clone(), output_index)?;
                let hidden = cache.activations.last().unwrap();
                if let Ok((loss, input_gradient, weight_gradient, bias_gradient)) = crate::gpu::time_distributed_loss_backward_device(&hidden.values, weights, biases, targets, hidden.height, *outputs, *inputs) {
                    let gradient = Tensor::new(1, hidden.height, hidden.width, input_gradient)?;
                    let mut gradients = self.backward_traversal_through(&cache, gradient, output_index)?;
                    gradients[output_index] = Some(LayerGradient { weights: GradientData::Device(weight_gradient), biases: GradientData::Device(bias_gradient) });
                    return Ok((loss, gradients));
                }
            }
        }

        let cache = self.forward_cached(input)?;
        let logits = cache.activations.last().unwrap();
        let vocabulary = logits.width;
        if logits.height != tokens.len() { return Err("model output does not have one row per token".into()); }

        #[cfg(feature = "gpu")]
        if crate::gpu::enabled() {
            if let Ok((loss, gradient_values)) = crate::gpu::softmax_cross_entropy(&logits.values, targets, tokens.len(), vocabulary) {
                let gradient = Tensor::new(1, logits.height, vocabulary, gradient_values)?;
                return Ok((loss, self.backward_traversal(&cache, gradient)?));
            }
        }

        let mut loss = 0.0;
        let mut gradient_values = vec![0.0; logits.values.len()];
        for position in 0..tokens.len() {
            let row = &logits.values[position * vocabulary..(position + 1) * vocabulary];
            let target = targets[position] as usize;
            if target >= vocabulary { return Err(format!("target token {target} is outside the vocabulary of {vocabulary}")); }
            let probabilities = softmax(row.to_vec());
            loss -= probabilities[target].max(1e-9).ln();
            for (column, probability) in probabilities.iter().enumerate() {
                gradient_values[position * vocabulary + column] = (probability - if column == target { 1.0 } else { 0.0 }) / tokens.len() as f32;
            }
        }
        let gradient = Tensor::new(1, logits.height, vocabulary, gradient_values)?;
        Ok((loss / tokens.len() as f32, self.backward_traversal(&cache, gradient)?))
    }

    fn backward_traversal(&self, cache: &ForwardCache, mut gradient: Tensor) -> Result<Vec<Option<LayerGradient>>, String> {
        self.backward_traversal_through(cache, gradient, self.layers.len())
    }

    fn backward_traversal_through(&self, cache: &ForwardCache, mut gradient: Tensor, layer_count: usize) -> Result<Vec<Option<LayerGradient>>, String> {
        let mut gradients: Vec<Option<LayerGradient>> = self.layers.iter().map(|_| None).collect();
        let mut bank_gradient = vec![0.0; cache.bank_lengths[layer_count]];
        for index in (0..layer_count).rev() {
            let input = &cache.activations[index];
            gradient = match &self.layers[index] {
                Layer::Dense { outputs, weights, .. } => { let (next, weight, bias) = dense_backward(input, *outputs, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::Flatten => Tensor::new(input.channels, input.height, input.width, gradient.values)?,
                Layer::Relu => relu_backward(input, &gradient)?,
                Layer::MaxPool2d { kernel_size } => max_pool2d_backward(input, *kernel_size, &gradient)?,
                Layer::Conv2d { out_channels, kernel_size, weights, .. } => { let (next, weight, bias) = conv2d_backward(input, *out_channels, *kernel_size, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::MemoryBank { inputs, slots, weights, .. } => { let (next, weight, bias) = memory_bank_backward(input, *inputs, *slots, weights, cache.bank_lengths[index], &gradient, &mut bank_gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                other => { let (next, weight, bias) = sequence_backward(other, input, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
            };
        }
        Ok(gradients)
    }

    /// Decoder-only transformer language model.
    pub fn language_model(vocab: usize, d_model: usize, heads: usize, ff_hidden: usize, blocks: usize, max_sequence: usize) -> Result<Self, String> {
        if vocab == 0 || d_model == 0 || heads == 0 || blocks == 0 || max_sequence == 0 { return Err("language model dimensions must all be positive".into()); }
        if d_model % heads != 0 { return Err("d_model must be a multiple of heads".into()); }
        let mut rng = rand::rng();
        let mut layers = vec![
            Layer::Embedding { vocab, d_model, weights: gaussian_weights(vocab * d_model, d_model, &mut rng).iter().map(|value| value * 0.5).collect(), biases: Vec::new() },
            Layer::PositionalEmbedding { max_sequence, d_model, weights: gaussian_weights(max_sequence * d_model, d_model, &mut rng).iter().map(|value| value * 0.1).collect(), biases: Vec::new() },
        ];
        for _ in 0..blocks {
            let shape = crate::transformer::BlockShape { sequence: 0, d_model, heads, ff_hidden };
            let mut weights = gaussian_weights(shape.weight_count(), d_model, &mut rng);
            // The trailing two blocks of the weight vector are the layer-norm gains.
            for value in weights[shape.weight_count() - 2 * d_model..].iter_mut() { *value = 1.0; }
            layers.push(Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases: vec![0.0; shape.bias_count()] });
        }
        layers.push(Layer::LayerNorm { size: d_model, weights: vec![1.0; d_model], biases: vec![0.0; d_model] });
        layers.push(Layer::TimeDistributedDense { inputs: d_model, outputs: vocab, weights: gaussian_weights(d_model * vocab, d_model, &mut rng), biases: vec![0.0; vocab] });
        Ok(Self::with_input_shape(1, 1, max_sequence, layers))
    }

    /// Trains on `(tokens, next_tokens)` pairs, returning the mean loss of the final epoch.
    pub fn train_language_model(&mut self, sequences: &[(Vec<u32>, Vec<u32>)], epochs: usize, learning_rate: f32, function: LearningFunction, interrupted: &AtomicBool, mut progress: impl FnMut(usize, f32)) -> Result<f32, String> {
        self.train_language_model_batched(sequences, epochs, 1, learning_rate, function, interrupted, progress)
    }

    /// Trains on fixed-length token sequences in microbatches.
    ///
    /// When CUDA is enabled, compatible gradients remain resident on the device while they are
    /// accumulated. This reduces optimizer launches and host/device transfers without changing
    /// the single-sequence behavior used by the existing API.
    pub fn train_language_model_batched(&mut self, sequences: &[(Vec<u32>, Vec<u32>)], epochs: usize, batch_size: usize, learning_rate: f32, function: LearningFunction, interrupted: &AtomicBool, mut progress: impl FnMut(usize, f32)) -> Result<f32, String> {
        if sequences.is_empty() || batch_size == 0 { return Err("language-model sequences and batch size must be non-zero".into()); }
        let mut optimizer = self.take_optimizer(function);
        let mut last = 0.0;
        for epoch in 1..=epochs {
            if interrupted.load(Ordering::Relaxed) { break; }
            let mut total = 0.0;
            for batch in sequences.chunks(batch_size) {
                let mut gradients: Option<Vec<Option<LayerGradient>>> = None;
                for (tokens, targets) in batch {
                    let (loss, sample) = self.language_model_step(tokens, targets)?;
                    total += loss;
                    if let Some(total) = gradients.as_mut() { add_gradients(total, &sample); } else { gradients = Some(sample); }
                }
                let mut gradients = gradients.expect("non-empty batch has gradients");
                if batch.len() > 1 { scale_gradients(&mut gradients, 1.0 / batch.len() as f32); }
                optimizer.step += 1;
                self.apply_gradients(&gradients, learning_rate, &mut optimizer);
            }
            last = total / sequences.len() as f32;
            progress(epoch, last);
        }
        self.flush_device_state(&mut optimizer);
        self.optimizer = Some(optimizer);
        Ok(last)
    }

    /// Logits for the final position only.
    ///
    /// Greedy decoding reads just the last row, so the output head is applied to that row alone
    /// instead of the whole sequence.
    fn last_row_logits(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let mut tensor = Tensor::new(1, 1, tokens.len(), tokens.iter().map(|token| *token as f32).collect())?;
        let final_index = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            if index == final_index {
                if let Layer::TimeDistributedDense { inputs, outputs, weights, biases } = layer {
                    if tensor.width != *inputs { return Err("output head width does not match its input".into()); }
                    let row = &tensor.values[(tensor.height - 1) * inputs..];
                    #[cfg(feature = "gpu")]
                    if crate::gpu::enabled() {
                        if let Ok(values) = crate::gpu::matmul_nt(row, weights, Some(biases), 1, *outputs, *inputs) { return Ok(values); }
                    }
                    return Ok((0..*outputs).map(|output| biases[output] + (0..*inputs).map(|column| row[column] * weights[output * inputs + column]).sum::<f32>()).collect());
                }
            }
            tensor = match layer {
                Layer::Conv2d { in_channels, out_channels, kernel_size, weights, biases } => conv2d(&tensor, *in_channels, *out_channels, *kernel_size, weights, biases)?,
                Layer::Relu => Tensor { values: tensor.values.iter().map(|value| value.max(0.0)).collect(), ..tensor },
                Layer::MaxPool2d { kernel_size } => max_pool2d(&tensor, *kernel_size)?,
                Layer::Flatten => Tensor { channels: 1, height: 1, width: tensor.values.len(), values: tensor.values },
                Layer::Dense { inputs, outputs, weights, biases } => dense(&tensor, *inputs, *outputs, weights, biases)?,
                other => sequence_forward(other, &tensor)?,
            };
        }
        Ok(tensor.values)
    }

    /// Greedy continuation of `prompt` for `count` further tokens.
    pub fn generate(&self, prompt: &[u32], count: usize, max_sequence: usize) -> Result<Vec<u32>, String> {
        self.generate_with(prompt, count, max_sequence, 0.0, 0)
    }

    /// Continues `prompt`, sampling from the top `top_k` logits at `temperature`.
    ///
    /// A temperature of zero or a `top_k` of one is greedy decoding.
    pub fn generate_with(&self, prompt: &[u32], count: usize, max_sequence: usize, temperature: f32, top_k: usize) -> Result<Vec<u32>, String> {
        let mut tokens = prompt.to_vec();
        let mut rng = rand::rng();
        for _ in 0..count {
            let window = tokens[tokens.len().saturating_sub(max_sequence)..].to_vec();
            let logits = self.last_row_logits(&window)?;
            if temperature <= 0.0 || top_k <= 1 {
                tokens.push(class(&logits) as u32);
                continue;
            }
            let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            let keep = top_k.min(ranked.len());
            ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            ranked.truncate(keep);
            let scaled: Vec<f32> = softmax(ranked.iter().map(|(_, value)| value / temperature).collect());
            let mut threshold: f32 = rng.random_range(0.0..1.0);
            let mut chosen = ranked[0].0;
            for ((token, _), probability) in ranked.iter().zip(&scaled) {
                threshold -= probability;
                if threshold <= 0.0 { chosen = *token; break; }
            }
            tokens.push(chosen as u32);
        }
        Ok(tokens)
    }

    /// Copies a pretrained matrix into the model's embedding, position, or output-head layer.
    ///
    /// Widths must match exactly; a larger source vocabulary is truncated to the model's, and a
    /// smaller one leaves the remaining rows untouched. Returns the number of rows copied.
    pub fn import_embeddings(&mut self, target: EmbeddingTarget, rows: usize, width: usize, values: &[f32]) -> Result<usize, String> {
        if values.len() != rows * width { return Err("embedding values do not match the stated shape".into()); }
        let (capacity, expected_width, weights) = match target {
            EmbeddingTarget::Token => self.layers.iter_mut().find_map(|layer| match layer {
                Layer::Embedding { vocab, d_model, weights, .. } => Some((*vocab, *d_model, weights)),
                _ => None,
            }).ok_or("model has no embedding layer")?,
            EmbeddingTarget::Position => self.layers.iter_mut().find_map(|layer| match layer {
                Layer::PositionalEmbedding { max_sequence, d_model, weights, .. } => Some((*max_sequence, *d_model, weights)),
                _ => None,
            }).ok_or("model has no positional embedding layer")?,
            EmbeddingTarget::OutputHead => self.layers.iter_mut().rev().find_map(|layer| match layer {
                Layer::TimeDistributedDense { inputs, outputs, weights, .. } => Some((*outputs, *inputs, weights)),
                _ => None,
            }).ok_or("model has no time-distributed output head")?,
        };
        if width != expected_width { return Err(format!("source width {width} does not match the target width {expected_width}")); }
        let copied = rows.min(capacity);
        weights[..copied * width].copy_from_slice(&values[..copied * width]);
        invalidate_device_cache();
        self.optimizer = None;
        Ok(copied)
    }

    /// Total number of trainable parameters.
    pub fn parameter_total(&self) -> usize { self.parameter_sizes().iter().sum() }

    /// Held-out loss and the number of positions whose highest-scoring token is correct.
    pub fn language_model_evaluate(&self, tokens: &[u32], targets: &[u32]) -> Result<(f32, usize), String> {
        if tokens.is_empty() || tokens.len() != targets.len() { return Err("token and target sequences must be equal and non-empty".into()); }
        let input = Tensor::new(1, 1, tokens.len(), tokens.iter().map(|token| *token as f32).collect())?;
        #[cfg(feature = "gpu")]
        if crate::gpu::enabled() {
            let output_index = self.layers.len().saturating_sub(1);
            if let Some(Layer::TimeDistributedDense { inputs, outputs, weights, biases }) = self.layers.last() {
                let cache = self.forward_cached_through(input.clone(), output_index)?;
                let hidden = cache.activations.last().unwrap();
                if let Ok(metrics) = crate::gpu::time_distributed_evaluate_device(&hidden.values, weights, biases, targets, hidden.height, *outputs, *inputs) {
                    return Ok(metrics);
                }
            }
        }
        let logits = self.forward_cached(input)?.activations.pop().unwrap();
        let vocabulary = logits.width;
        let mut loss = 0.0;
        let mut correct = 0;
        for position in 0..tokens.len() {
            let row = &logits.values[position * vocabulary..(position + 1) * vocabulary];
            let target = targets[position] as usize;
            if target >= vocabulary { return Err(format!("target token {target} is outside the vocabulary of {vocabulary}")); }
            loss -= softmax(row.to_vec())[target].max(1e-9).ln();
            if class(row) == target { correct += 1; }
        }
        Ok((loss / tokens.len() as f32, correct))
    }

    /// Logits at the final position, exposed for growth tests.
    pub(crate) fn last_row_logits_for_test(&self, tokens: &[u32]) -> Result<Vec<f32>, String> { self.last_row_logits(tokens) }

    /// Drops optimizer state and cached device buffers after the layer shapes change.
    pub(crate) fn invalidate_after_growth(&mut self) {
        invalidate_device_cache();
        #[cfg(feature = "gpu")]
        crate::gpu::clear_moment_buffers();
        self.optimizer = None;
    }

    /// Longest sequence the positional embedding can cover.
    pub fn max_sequence(&self) -> usize {
        self.layers.iter().find_map(|layer| match layer { Layer::PositionalEmbedding { max_sequence, .. } => Some(*max_sequence), _ => None }).unwrap_or(self.input_width)
    }

    pub fn accuracy(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<f32, String> {
        if inputs.is_empty() || inputs.len() != targets.len() { return Err("CNN inputs and targets must have equal non-zero length".into()); }
        let correct = inputs.iter().zip(targets).filter(|(input, target)| self.forward(Tensor::new(self.input_channels, self.input_height, self.input_width, (*input).clone()).unwrap()).map(|prediction| class(&prediction) == class(target)).unwrap_or(false)).count();
        Ok(correct as f32 / inputs.len() as f32)
    }

    fn backward(&self, input: &[f32], target: &[f32]) -> Result<Vec<Option<LayerGradient>>, String> {
        let cache = self.forward_cached(Tensor::new(self.input_channels, self.input_height, self.input_width, input.to_vec())?)?;
        let prediction = softmax(cache.activations.last().unwrap().values.clone());
        if prediction.len() != target.len() { return Err("CNN target width does not match output".into()); }
        let mut gradient = Tensor::new(1, 1, prediction.len(), prediction.iter().zip(target).map(|(value, target)| value - target).collect())?;
        let mut gradients: Vec<Option<LayerGradient>> = self.layers.iter().map(|_| None).collect();
        let mut bank_gradient = vec![0.0; *cache.bank_lengths.last().unwrap()];
        for index in (0..self.layers.len()).rev() {
            let input = &cache.activations[index];
            gradient = match &self.layers[index] {
                Layer::Dense { outputs, weights, .. } => { let (next, weight, bias) = dense_backward(input, *outputs, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::Flatten => Tensor::new(input.channels, input.height, input.width, gradient.values)?,
                Layer::Relu => relu_backward(input, &gradient)?,
                Layer::MaxPool2d { kernel_size } => max_pool2d_backward(input, *kernel_size, &gradient)?,
                Layer::Conv2d { out_channels, kernel_size, weights, .. } => { let (next, weight, bias) = conv2d_backward(input, *out_channels, *kernel_size, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::MemoryBank { inputs, slots, weights, .. } => { let (next, weight, bias) = memory_bank_backward(input, *inputs, *slots, weights, cache.bank_lengths[index], &gradient, &mut bank_gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                other => { let (next, weight, bias) = sequence_backward(other, input, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
            };
        }
        Ok(gradients)
    }

    #[cfg(feature = "gpu")]
    fn cuda_backward(&self, input: &[f32], target: &[f32]) -> Result<Vec<Option<LayerGradient>>, String> {
        let cache = self.forward_cached(Tensor::new(self.input_channels, self.input_height, self.input_width, input.to_vec())?)?;
        let prediction = softmax(cache.activations.last().unwrap().values.clone());
        if prediction.len() != target.len() { return Err("CNN target width does not match output".into()); }
        let mut gradient = Tensor::new(1, 1, prediction.len(), prediction.iter().zip(target).map(|(value, target)| value - target).collect())?;
        let mut gradients: Vec<Option<LayerGradient>> = self.layers.iter().map(|_| None).collect();
        let mut bank_gradient = vec![0.0; *cache.bank_lengths.last().unwrap()];
        for index in (0..self.layers.len()).rev() {
            let input = &cache.activations[index];
            gradient = match &self.layers[index] {
                Layer::Dense { outputs, weights, .. } => { let (next, weight, bias) = crate::gpu::cnn_dense_backward(input, *outputs, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::Flatten => Tensor::new(input.channels, input.height, input.width, gradient.values)?,
                Layer::Relu => crate::gpu::cnn_relu_backward(input, &gradient)?,
                Layer::MaxPool2d { kernel_size } => crate::gpu::cnn_max_pool2d_backward(input, *kernel_size, &gradient)?,
                Layer::Conv2d { out_channels, kernel_size, weights, .. } => { let (next, weight, bias) = crate::gpu::cnn_conv2d_backward(input, *out_channels, *kernel_size, weights, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                Layer::MemoryBank { inputs, slots, weights, .. } => { let (next, weight, bias) = memory_bank_backward(input, *inputs, *slots, weights, cache.bank_lengths[index], &gradient, &mut bank_gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
                other => { let (next, weight, bias) = sequence_backward(other, input, &gradient)?; gradients[index] = Some(LayerGradient { weights: weight.into(), biases: bias.into() }); next }
            };
        }
        Ok(gradients)
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn cuda_sample_gradients(&self, input: &[f32], target: &[f32]) -> Result<Vec<Option<LayerGradient>>, String> { self.cuda_backward(input, target) }

    pub(crate) fn cpu_sample_gradients(&self, input: &[f32], target: &[f32]) -> Result<Vec<Option<LayerGradient>>, String> { self.backward(input, target) }

    /// Plain fully connected view used by the model-surgery commands.
    fn dense_chain(&self) -> Result<DenseChain, String> {
        let mut connections = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Flatten | Layer::Relu => {}
                Layer::Dense { weights, biases, .. } => connections.push(Connection { weights: weights.clone(), biases: biases.clone() }),
                _ => return Err("model surgery requires a plain fully connected model; convolution, pooling, and memory bank layers cannot be reshaped".into()),
            }
        }
        if connections.is_empty() { return Err("model has no dense layers to reshape".into()); }
        Ok(DenseChain { sizes: self.layer_sizes(), connections })
    }

    fn set_dense_chain(&mut self, chain: DenseChain) {
        let last = chain.connections.len() - 1;
        let mut layers = vec![Layer::Flatten];
        for (index, connection) in chain.connections.into_iter().enumerate() {
            layers.push(Layer::Dense { inputs: chain.sizes[index], outputs: chain.sizes[index + 1], weights: connection.weights, biases: connection.biases });
            if index != last { layers.push(Layer::Relu); }
        }
        self.layers = layers;
        invalidate_device_cache();
        self.optimizer = None;
    }

    pub fn insert_layer(&mut self, insert_after: usize, size: usize, initialization: LayerInit) -> Result<(), String> {
        let mut chain = self.dense_chain()?;
        if size == 0 { return Err("layer size must be greater than zero".into()); }
        if insert_after == 0 || insert_after >= chain.sizes.len() - 1 { return Err("insert-at must name a hidden layer before the output layer; input and output layers cannot be inserted".into()); }
        let input_size = chain.sizes[insert_after];
        let output_size = chain.sizes[insert_after + 1];
        let old = chain.connections.remove(insert_after);
        let (input_layer, output_layer) = match initialization {
            LayerInit::Passthrough => passthrough_connections(input_size, size, output_size, old),
            LayerInit::Gaussian => gaussian_connections(input_size, size, output_size, old.biases),
            LayerInit::Copy { source_layer } => {
                let source = chain.connections.get(source_layer).ok_or("copy source layer does not exist")?;
                if source.weights.len() != size * input_size || source.biases.len() != size { return Err("copy source must have the same input width and requested layer size".into()); }
                if old.weights.len() != output_size * size { return Err("copy method requires the requested layer size to match the replaced connection input width".into()); }
                (source.clone(), old)
            }
            LayerInit::Values(values) => values_connections(input_size, size, output_size, old.biases, values)?,
        };
        chain.connections.insert(insert_after, input_layer);
        chain.connections.insert(insert_after + 1, output_layer);
        chain.sizes.insert(insert_after + 1, size);
        self.set_dense_chain(chain);
        Ok(())
    }

    pub fn add_neurons(&mut self, layer_index: usize, insert_at: usize, count: usize, gaussian: bool) -> Result<(), String> {
        let mut chain = self.dense_chain()?;
        chain.validate_hidden_layer(layer_index)?;
        let old_size = chain.sizes[layer_index];
        if count == 0 || insert_at > old_size { return Err("neuron count must be positive and insert-at must be within the layer".into()); }
        let previous_size = chain.sizes[layer_index - 1];
        let next_size = chain.sizes[layer_index + 1];
        let incoming = &chain.connections[layer_index - 1];
        let outgoing = &chain.connections[layer_index];
        let mut new_incoming = Connection { weights: vec![0.0; (old_size + count) * previous_size], biases: vec![0.0; old_size + count] };
        let mut new_outgoing = Connection { weights: vec![0.0; next_size * (old_size + count)], biases: outgoing.biases.clone() };
        for new_neuron in 0..old_size + count {
            if (insert_at..insert_at + count).contains(&new_neuron) { continue; }
            let old_neuron = if new_neuron < insert_at { new_neuron } else { new_neuron - count };
            new_incoming.weights[new_neuron * previous_size..(new_neuron + 1) * previous_size].copy_from_slice(&incoming.weights[old_neuron * previous_size..(old_neuron + 1) * previous_size]);
            new_incoming.biases[new_neuron] = incoming.biases[old_neuron];
            for next in 0..next_size { new_outgoing.weights[next * (old_size + count) + new_neuron] = outgoing.weights[next * old_size + old_neuron]; }
        }
        if gaussian {
            let mut rng = rand::rng();
            let input_scale = (2.0 / previous_size as f32).sqrt();
            let output_scale = (2.0 / (old_size + count) as f32).sqrt();
            for neuron in insert_at..insert_at + count {
                for input in 0..previous_size { let sample: f32 = StandardNormal.sample(&mut rng); new_incoming.weights[neuron * previous_size + input] = sample * input_scale; }
                for next in 0..next_size { let sample: f32 = StandardNormal.sample(&mut rng); new_outgoing.weights[next * (old_size + count) + neuron] = sample * output_scale; }
            }
        }
        chain.connections[layer_index - 1] = new_incoming;
        chain.connections[layer_index] = new_outgoing;
        chain.sizes[layer_index] += count;
        self.set_dense_chain(chain);
        Ok(())
    }

    pub fn remove_neurons(&mut self, layer_index: usize, indexes: &[usize]) -> Result<(), String> {
        let mut chain = self.dense_chain()?;
        chain.validate_hidden_layer(layer_index)?;
        let old_size = chain.sizes[layer_index];
        if indexes.is_empty() { return Err("provide at least one neuron index to remove".into()); }
        let mut remove = indexes.to_vec();
        remove.sort_unstable();
        remove.dedup();
        if remove.len() >= old_size || remove.iter().any(|&index| index >= old_size) { return Err("neuron indexes must be valid and leave at least one neuron".into()); }
        let keep: Vec<_> = (0..old_size).filter(|index| !remove.contains(index)).collect();
        let previous_size = chain.sizes[layer_index - 1];
        let next_size = chain.sizes[layer_index + 1];
        let incoming = &chain.connections[layer_index - 1];
        let outgoing = &chain.connections[layer_index];
        let mut new_incoming = Connection { weights: Vec::with_capacity(keep.len() * previous_size), biases: Vec::with_capacity(keep.len()) };
        let mut new_outgoing = Connection { weights: vec![0.0; next_size * keep.len()], biases: outgoing.biases.clone() };
        for (new_neuron, old_neuron) in keep.iter().copied().enumerate() {
            new_incoming.weights.extend_from_slice(&incoming.weights[old_neuron * previous_size..(old_neuron + 1) * previous_size]);
            new_incoming.biases.push(incoming.biases[old_neuron]);
            for next in 0..next_size { new_outgoing.weights[next * keep.len() + new_neuron] = outgoing.weights[next * old_size + old_neuron]; }
        }
        chain.connections[layer_index - 1] = new_incoming;
        chain.connections[layer_index] = new_outgoing;
        chain.sizes[layer_index] = keep.len();
        self.set_dense_chain(chain);
        Ok(())
    }

    pub fn remove_layer(&mut self, layer_index: usize) -> Result<(), String> {
        let mut chain = self.dense_chain()?;
        chain.validate_hidden_layer(layer_index)?;
        let previous_size = chain.sizes[layer_index - 1];
        let removed_size = chain.sizes[layer_index];
        let next_size = chain.sizes[layer_index + 1];
        let incoming = chain.connections.remove(layer_index - 1);
        let outgoing = chain.connections.remove(layer_index - 1);
        let mut composed = Connection { weights: vec![0.0; next_size * previous_size], biases: outgoing.biases };
        for next in 0..next_size {
            composed.biases[next] += (0..removed_size).map(|removed| outgoing.weights[next * removed_size + removed] * incoming.biases[removed]).sum::<f32>();
            for previous in 0..previous_size {
                composed.weights[next * previous_size + previous] = (0..removed_size).map(|removed| outgoing.weights[next * removed_size + removed] * incoming.weights[removed * previous_size + previous]).sum();
            }
        }
        chain.connections.insert(layer_index - 1, composed);
        chain.sizes.remove(layer_index);
        self.set_dense_chain(chain);
        Ok(())
    }

    fn zero_gradients(&self) -> Vec<Option<LayerGradient>> { self.layers.iter().map(|layer| match layer { Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. } => Some(LayerGradient { weights: vec![0.0; weights.len()].into(), biases: vec![0.0; biases.len()].into() }), _ => None }).collect() }    fn parameter_sizes(&self) -> Vec<usize> { self.layers.iter().flat_map(|layer| match layer { Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. } => vec![weights.len(), biases.len()], _ => Vec::new() }).collect() }
    fn apply_gradients(&mut self, gradients: &[Option<LayerGradient>], learning_rate: f32, optimizer: &mut ModelOptimizer) {
        #[cfg(feature = "gpu")]
        if crate::gpu::enabled() && matches!(optimizer.function, LearningFunction::Adam | LearningFunction::AdamW { .. }) && self.apply_gradients_on_device(gradients, learning_rate, optimizer).is_ok() {
            // Host and device parameters were updated together, so the cache stays valid.
            return;
        }
        invalidate_device_cache();
        let mut parameter_index = 0;
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) { if let (Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. }, Some(gradient)) = (layer, gradient) {
            update_slice(weights, &gradient.weights.to_host(), learning_rate, optimizer.function, &mut optimizer.first[parameter_index], &mut optimizer.second[parameter_index], optimizer.step); parameter_index += 1;
            update_slice(biases, &gradient.biases.to_host(), learning_rate, optimizer.function, &mut optimizer.first[parameter_index], &mut optimizer.second[parameter_index], optimizer.step); parameter_index += 1;
        } }
    }
}

/// Layer topology without weights; stored as GGUF metadata so any graph shape can be rebuilt.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LayerSpec {
    Conv2d { in_channels: usize, out_channels: usize, kernel_size: usize },
    Relu,
    MaxPool2d { kernel_size: usize },
    Flatten,
    Dense { inputs: usize, outputs: usize },
    MemoryBank { inputs: usize, slots: usize },
    Embedding { vocab: usize, d_model: usize },
    PositionalEmbedding { max_sequence: usize, d_model: usize },
    TransformerBlock { d_model: usize, heads: usize, ff_hidden: usize },
    LayerNorm { size: usize },
    TimeDistributedDense { inputs: usize, outputs: usize },
}

impl LayerSpec {
    fn of(layer: &Layer) -> Self {
        match layer {
            Layer::Conv2d { in_channels, out_channels, kernel_size, .. } => LayerSpec::Conv2d { in_channels: *in_channels, out_channels: *out_channels, kernel_size: *kernel_size },
            Layer::Relu => LayerSpec::Relu,
            Layer::MaxPool2d { kernel_size } => LayerSpec::MaxPool2d { kernel_size: *kernel_size },
            Layer::Flatten => LayerSpec::Flatten,
            Layer::Dense { inputs, outputs, .. } => LayerSpec::Dense { inputs: *inputs, outputs: *outputs },
            Layer::MemoryBank { inputs, slots, .. } => LayerSpec::MemoryBank { inputs: *inputs, slots: *slots },
            Layer::Embedding { vocab, d_model, .. } => LayerSpec::Embedding { vocab: *vocab, d_model: *d_model },
            Layer::PositionalEmbedding { max_sequence, d_model, .. } => LayerSpec::PositionalEmbedding { max_sequence: *max_sequence, d_model: *d_model },
            Layer::TransformerBlock { d_model, heads, ff_hidden, .. } => LayerSpec::TransformerBlock { d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden },
            Layer::LayerNorm { size, .. } => LayerSpec::LayerNorm { size: *size },
            Layer::TimeDistributedDense { inputs, outputs, .. } => LayerSpec::TimeDistributedDense { inputs: *inputs, outputs: *outputs },
        }
    }

    fn instantiate(&self) -> Layer {
        match self {
            LayerSpec::Conv2d { in_channels, out_channels, kernel_size } => Layer::Conv2d { in_channels: *in_channels, out_channels: *out_channels, kernel_size: *kernel_size, weights: vec![0.0; out_channels * in_channels * kernel_size * kernel_size], biases: vec![0.0; *out_channels] },
            LayerSpec::Relu => Layer::Relu,
            LayerSpec::MaxPool2d { kernel_size } => Layer::MaxPool2d { kernel_size: *kernel_size },
            LayerSpec::Flatten => Layer::Flatten,
            LayerSpec::Dense { inputs, outputs } => Layer::Dense { inputs: *inputs, outputs: *outputs, weights: vec![0.0; inputs * outputs], biases: vec![0.0; *outputs] },
            LayerSpec::MemoryBank { inputs, slots } => Layer::MemoryBank { inputs: *inputs, slots: *slots, weights: vec![0.0; inputs * slots], biases: vec![0.0; *slots] },
            LayerSpec::Embedding { vocab, d_model } => Layer::Embedding { vocab: *vocab, d_model: *d_model, weights: vec![0.0; vocab * d_model], biases: Vec::new() },
            LayerSpec::PositionalEmbedding { max_sequence, d_model } => Layer::PositionalEmbedding { max_sequence: *max_sequence, d_model: *d_model, weights: vec![0.0; max_sequence * d_model], biases: Vec::new() },
            LayerSpec::TransformerBlock { d_model, heads, ff_hidden } => {
                let shape = crate::transformer::BlockShape { sequence: 0, d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden };
                Layer::TransformerBlock { d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden, weights: vec![0.0; shape.weight_count()], biases: vec![0.0; shape.bias_count()] }
            }
            LayerSpec::LayerNorm { size } => Layer::LayerNorm { size: *size, weights: vec![1.0; *size], biases: vec![0.0; *size] },
            LayerSpec::TimeDistributedDense { inputs, outputs } => Layer::TimeDistributedDense { inputs: *inputs, outputs: *outputs, weights: vec![0.0; inputs * outputs], biases: vec![0.0; *outputs] },
        }
    }
}

/// Weight and bias slices of every parameterized layer, in graph order.
fn parameters(layers: &[Layer]) -> Vec<(usize, &Vec<f32>, &Vec<f32>)> {
    layers.iter().enumerate().filter_map(|(index, layer)| match layer {
        Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. } => Some((index, weights, biases)),
        _ => None,
    }).collect()
}

fn optimizer_path(path: &Path) -> PathBuf { PathBuf::from(format!("{}.optimizer.bin", path.display())) }
fn legacy_optimizer_path(path: &Path) -> PathBuf { PathBuf::from(format!("{}.optimizer.json", path.display())) }

fn temporary_path(path: &Path) -> PathBuf {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("model");
    path.with_file_name(format!(".{name}.partial"))
}

/// Moment buffers are far too large to store as JSON text, so they use a flat binary layout.
fn encode_optimizer(state: &ModelOptimizer) -> Result<Vec<u8>, String> {
    let mut bytes = b"NNOPT1".to_vec();
    gguf::u32put(&mut bytes, state.step);
    let function = serde_json::to_vec(&state.function).map_err(|error| error.to_string())?;
    gguf::u32put(&mut bytes, function.len() as u32);
    bytes.extend(function);
    gguf::u64put(&mut bytes, state.first.len() as u64);
    for group in &state.first { gguf::u64put(&mut bytes, group.len() as u64); }
    for group in state.first.iter().chain(&state.second) { gguf::f32s(&mut bytes, group); }
    Ok(bytes)
}

fn decode_optimizer(bytes: &[u8]) -> Result<ModelOptimizer, String> {
    let mut at = 0;
    if gguf::take(bytes, &mut at, 6)? != b"NNOPT1" { return Err("not an optimizer checkpoint".into()); }
    let step = gguf::get_u32(bytes, &mut at)?;
    let function_length = gguf::get_u32(bytes, &mut at)? as usize;
    let function = serde_json::from_slice(gguf::take(bytes, &mut at, function_length)?).map_err(|error| error.to_string())?;
    let groups = gguf::get_u64(bytes, &mut at)? as usize;
    let sizes: Vec<usize> = (0..groups).map(|_| gguf::get_u64(bytes, &mut at).map(|size| size as usize)).collect::<Result<_, _>>()?;
    let mut read_groups = || sizes.iter().map(|size| {
        let raw = gguf::take(bytes, &mut at, size * 4)?;
        Ok(raw.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect())
    }).collect::<Result<Vec<Vec<f32>>, String>>();
    let first = read_groups()?;
    let second = read_groups()?;
    Ok(ModelOptimizer { function, step, first, second })
}

pub fn save_model(model: &Model, path: &Path) -> Result<(), String> {
    let parameters = parameters(&model.layers);
    let specs: Vec<LayerSpec> = model.layers.iter().map(LayerSpec::of).collect();
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    gguf::u32put(&mut header, 3);
    gguf::u64put(&mut header, (parameters.len() * 2) as u64);
    gguf::u64put(&mut header, 4);
    gguf::meta_str(&mut header, "general.architecture", "neuralnet");
    gguf::meta_u32(&mut header, "general.alignment", gguf::ALIGNMENT as u32);
    gguf::meta_u32s(&mut header, "neuralnet.input_shape", &[model.input_channels as u32, model.input_height as u32, model.input_width as u32]);
    gguf::meta_str(&mut header, "neuralnet.layers", &serde_json::to_string(&specs).map_err(|error| error.to_string())?);
    let mut offset = 0;
    for (ordinal, (_, weights, biases)) in parameters.iter().enumerate() {
        gguf::tensor(&mut header, &format!("layer.{ordinal}.weight"), &[weights.len() as u64], offset);
        offset = gguf::aligned(offset + weights.len() * 4);
        gguf::tensor(&mut header, &format!("layer.{ordinal}.bias"), &[biases.len() as u64], offset);
        offset = gguf::aligned(offset + biases.len() * 4);
    }
    let mut file = header;
    file.resize(gguf::aligned(file.len()), 0);
    for (_, weights, biases) in &parameters {
        gguf::f32s(&mut file, weights);
        file.resize(gguf::aligned(file.len()), 0);
        gguf::f32s(&mut file, biases);
        file.resize(gguf::aligned(file.len()), 0);
    }
    let temporary_model = temporary_path(path);
    fs::write(&temporary_model, file).map_err(|error| error.to_string())?;
    let checkpoint = optimizer_path(path);
    match &model.optimizer {
        Some(state) => {
            let temporary_optimizer = temporary_path(&checkpoint);
            fs::write(&temporary_optimizer, encode_optimizer(state)?).map_err(|error| error.to_string())?;
            if checkpoint.exists() { fs::remove_file(&checkpoint).map_err(|error| error.to_string())?; }
            fs::rename(&temporary_model, path).map_err(|error| error.to_string())?;
            fs::rename(temporary_optimizer, checkpoint).map_err(|error| error.to_string())?;
            return Ok(());
        }
        None if checkpoint.exists() => fs::remove_file(&checkpoint).map_err(|error| error.to_string())?,
        None => {}
    }
    fs::rename(temporary_model, path).map_err(|error| error.to_string())?;
    Ok(())
}

pub fn load_model(path: &Path) -> Result<Model, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    // Older CNN checkpoints were plain serde_json; keep reading them.
    let mut model = if bytes.starts_with(b"GGUF") { load_gguf(&bytes)? } else { serde_json::from_slice(&bytes).map_err(|error| error.to_string())? };
    if model.optimizer.is_none() {
        model.optimizer = fs::read(optimizer_path(path)).ok().and_then(|bytes| decode_optimizer(&bytes).ok())
            .or_else(|| fs::read(legacy_optimizer_path(path)).ok().and_then(|bytes| serde_json::from_slice::<ModelOptimizer>(&bytes).ok()))
            .filter(|state| state.matches(&model));
    }
    Ok(model)
}

fn load_gguf(bytes: &[u8]) -> Result<Model, String> {
    let mut specs: Option<Vec<LayerSpec>> = None;
    let mut input_shape: Option<Vec<usize>> = None;
    let mut legacy_sizes: Option<Vec<usize>> = None;
    let (info, data_start) = gguf::read_header(bytes, |key, kind, bytes, at| match (key, kind) {
        ("neuralnet.layers", 8) => { specs = Some(serde_json::from_str(&gguf::get_str(bytes, at)?).map_err(|error| error.to_string())?); Ok(true) }
        ("neuralnet.input_shape", 9) => { input_shape = Some(gguf::get_u32_array(bytes, at)?); Ok(true) }
        ("neuralnet.layer_sizes", 9) => { legacy_sizes = Some(gguf::get_u32_array(bytes, at)?); Ok(true) }
        _ => Ok(false),
    })?;

    let (specs, channels, height, width) = match (specs, legacy_sizes) {
        (Some(specs), _) => {
            let shape = input_shape.ok_or("GGUF model lacks neuralnet.input_shape")?;
            if shape.len() != 3 { return Err("neuralnet.input_shape must have three entries".into()); }
            (specs, shape[0], shape[1], shape[2])
        }
        // Legacy dense models stored only layer widths; they are a Flatten/Dense/ReLU chain.
        (None, Some(sizes)) => {
            if sizes.len() < 2 { return Err("GGUF model needs at least two layer sizes".into()); }
            let mut specs = vec![LayerSpec::Flatten];
            for (index, pair) in sizes.windows(2).enumerate() {
                specs.push(LayerSpec::Dense { inputs: pair[0], outputs: pair[1] });
                if index + 2 < sizes.len() { specs.push(LayerSpec::Relu); }
            }
            (specs, 1, 1, sizes[0])
        }
        (None, None) => return Err("GGUF model lacks neuralnet.layers".into()),
    };

    let mut layers: Vec<Layer> = specs.iter().map(LayerSpec::instantiate).collect();
    let mut ordinal = 0;
    for layer in &mut layers {
        if let Layer::Conv2d { weights, biases, .. } | Layer::Dense { weights, biases, .. } | Layer::MemoryBank { weights, biases, .. } | Layer::Embedding { weights, biases, .. } | Layer::PositionalEmbedding { weights, biases, .. } | Layer::TransformerBlock { weights, biases, .. } | Layer::LayerNorm { weights, biases, .. } | Layer::TimeDistributedDense { weights, biases, .. } = layer {
            *weights = gguf::tensor_values(bytes, data_start, &info, &format!("layer.{ordinal}.weight"), weights.len())?;
            *biases = gguf::tensor_values(bytes, data_start, &info, &format!("layer.{ordinal}.bias"), biases.len())?;
            ordinal += 1;
        }
    }
    Ok(Model::with_input_shape(channels, height, width, layers))
}

fn add_gradients(total: &mut [Option<LayerGradient>], sample: &[Option<LayerGradient>]) { for (total, sample) in total.iter_mut().zip(sample) { if let (Some(total), Some(sample)) = (total, sample) { let _ = total.weights.add_assign(&sample.weights); let _ = total.biases.add_assign(&sample.biases); } } }
fn scale_gradients(gradients: &mut [Option<LayerGradient>], scale: f32) { for gradient in gradients.iter_mut().flatten() { let _ = gradient.weights.scale(scale); let _ = gradient.biases.scale(scale); } }
fn update_slice(parameters: &mut [f32], gradients: &[f32], rate: f32, function: LearningFunction, first: &mut [f32], second: &mut [f32], step: u32) { for ((parameter, gradient), (first, second)) in parameters.iter_mut().zip(gradients).zip(first.iter_mut().zip(second)) { function.update(parameter, *gradient, rate, first, second, step); } }
fn class(values: &[f32]) -> usize { values.iter().enumerate().max_by(|(_, left), (_, right)| left.total_cmp(right)).map(|(index, _)| index).unwrap_or(0) }

pub(crate) fn conv2d(input: &Tensor, in_channels: usize, out_channels: usize, kernel: usize, weights: &[f32], biases: &[f32]) -> Result<Tensor, String> {
    if input.channels != in_channels || input.height < kernel || input.width < kernel || weights.len() != out_channels * in_channels * kernel * kernel || biases.len() != out_channels { return Err("invalid convolution shape".into()); }
    let height = input.height - kernel + 1; let width = input.width - kernel + 1;
    let mut values = vec![0.0; out_channels * height * width];
    for output in 0..out_channels { for row in 0..height { for column in 0..width {
        let mut sum = biases[output];
        for channel in 0..in_channels { for ky in 0..kernel { for kx in 0..kernel {
            sum += input.values[channel * input.height * input.width + (row + ky) * input.width + column + kx] * weights[((output * in_channels + channel) * kernel + ky) * kernel + kx];
        } } }
        values[output * height * width + row * width + column] = sum;
    } } }
    Tensor::new(out_channels, height, width, values)
}

pub(crate) fn max_pool2d(input: &Tensor, kernel: usize) -> Result<Tensor, String> {
    if kernel == 0 || input.height < kernel || input.width < kernel { return Err("max-pool kernel must fit within the tensor dimensions".into()); }
    let height = input.height / kernel; let width = input.width / kernel; let mut values = vec![0.0; input.channels * height * width];
    for channel in 0..input.channels { for row in 0..height { for column in 0..width {
        values[channel * height * width + row * width + column] = (0..kernel * kernel).map(|index| input.values[channel * input.height * input.width + (row * kernel + index / kernel) * input.width + column * kernel + index % kernel]).fold(f32::NEG_INFINITY, f32::max);
    } } }
    Tensor::new(input.channels, height, width, values)
}

pub(crate) fn relu_backward(input: &Tensor, gradient: &Tensor) -> Result<Tensor, String> {
    if input.channels != gradient.channels || input.height != gradient.height || input.width != gradient.width { return Err("ReLU gradient shape does not match input".into()); }
    Tensor::new(input.channels, input.height, input.width, input.values.iter().zip(&gradient.values).map(|(value, gradient)| if *value > 0.0 { *gradient } else { 0.0 }).collect())
}

pub(crate) fn max_pool2d_backward(input: &Tensor, kernel: usize, gradient: &Tensor) -> Result<Tensor, String> {
    let pooled = max_pool2d(input, kernel)?;
    if pooled.channels != gradient.channels || pooled.height != gradient.height || pooled.width != gradient.width { return Err("max-pool gradient shape does not match output".into()); }
    let mut result = vec![0.0; input.values.len()];
    for channel in 0..pooled.channels { for row in 0..pooled.height { for column in 0..pooled.width {
        let mut maximum = f32::NEG_INFINITY; let mut maximum_index = 0;
        for ky in 0..kernel { for kx in 0..kernel { let index = channel * input.height * input.width + (row * kernel + ky) * input.width + column * kernel + kx; if input.values[index] > maximum { maximum = input.values[index]; maximum_index = index; } } }
        result[maximum_index] = gradient.values[channel * pooled.height * pooled.width + row * pooled.width + column];
    } } }
    Tensor::new(input.channels, input.height, input.width, result)
}

pub(crate) fn dense_backward(input: &Tensor, outputs: usize, weights: &[f32], gradient: &Tensor) -> Result<(Tensor, Vec<f32>, Vec<f32>), String> {
    let inputs = input.values.len();
    if weights.len() != inputs * outputs || gradient.values.len() != outputs { return Err("invalid dense backward shape".into()); }
    let mut input_gradient = vec![0.0; inputs]; let mut weight_gradient = vec![0.0; weights.len()];
    for output in 0..outputs { for index in 0..inputs { weight_gradient[output * inputs + index] = gradient.values[output] * input.values[index]; input_gradient[index] += weights[output * inputs + index] * gradient.values[output]; } }
    Ok((Tensor::new(input.channels, input.height, input.width, input_gradient)?, weight_gradient, gradient.values.clone()))
}

pub(crate) fn conv2d_backward(input: &Tensor, out_channels: usize, kernel: usize, weights: &[f32], gradient: &Tensor) -> Result<(Tensor, Vec<f32>, Vec<f32>), String> {
    let output = conv2d(input, input.channels, out_channels, kernel, weights, &vec![0.0; out_channels])?;
    if output.channels != gradient.channels || output.height != gradient.height || output.width != gradient.width { return Err("convolution gradient shape does not match output".into()); }
    let mut input_gradient = vec![0.0; input.values.len()]; let mut weight_gradient = vec![0.0; weights.len()]; let mut bias_gradient = vec![0.0; out_channels];
    for output_channel in 0..out_channels { for row in 0..output.height { for column in 0..output.width { let delta = gradient.values[output_channel * output.height * output.width + row * output.width + column]; bias_gradient[output_channel] += delta;
        for channel in 0..input.channels { for ky in 0..kernel { for kx in 0..kernel { let input_index = channel * input.height * input.width + (row + ky) * input.width + column + kx; let weight_index = ((output_channel * input.channels + channel) * kernel + ky) * kernel + kx; weight_gradient[weight_index] += delta * input.values[input_index]; input_gradient[input_index] += delta * weights[weight_index]; } } }
    } } }
    Ok((Tensor::new(input.channels, input.height, input.width, input_gradient)?, weight_gradient, bias_gradient))
}

#[derive(Clone, Copy, PartialEq)]
pub enum EmbeddingTarget { Token, Position, OutputHead }

pub enum LayerInit {
    Passthrough,
    Gaussian,
    Copy { source_layer: usize },
    Values(LayerValues),
}

#[derive(Debug, Deserialize)]
pub struct LayerValues {
    pub input_weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub output_weights: Vec<f32>,
    #[serde(default)]
    pub output_biases: Option<Vec<f32>>,
}

#[derive(Clone)]
struct Connection { weights: Vec<f32>, biases: Vec<f32> }

struct DenseChain { sizes: Vec<usize>, connections: Vec<Connection> }

impl DenseChain {
    fn validate_hidden_layer(&self, layer_index: usize) -> Result<(), String> {
        if layer_index == 0 || layer_index + 1 >= self.sizes.len() { Err("layer index must identify a hidden layer; input and output layers cannot be changed".into()) } else { Ok(()) }
    }
}

fn passthrough_connections(input_size: usize, new_size: usize, output_size: usize, old: Connection) -> (Connection, Connection) {
    let mut input_weights = vec![0.0; new_size * input_size];
    let mut output_weights = vec![0.0; output_size * new_size];
    for source in 0..input_size {
        let start = source * new_size / input_size;
        let end = ((source + 1) * new_size / input_size).max(start + 1).min(new_size);
        let copies = end - start;
        for unit in start..end {
            input_weights[unit * input_size + source] = 1.0;
            for output in 0..output_size { output_weights[output * new_size + unit] = old.weights[output * input_size + source] / copies as f32; }
        }
    }
    if new_size < input_size {
        input_weights.fill(0.0);
        output_weights.fill(0.0);
        for unit in 0..new_size {
            let start = unit * input_size / new_size;
            let end = ((unit + 1) * input_size / new_size).max(start + 1).min(input_size);
            let group = end - start;
            for source in start..end {
                input_weights[unit * input_size + source] = 1.0 / group as f32;
                for output in 0..output_size { output_weights[output * new_size + unit] += old.weights[output * input_size + source]; }
            }
        }
    }
    (Connection { weights: input_weights, biases: vec![0.0; new_size] }, Connection { weights: output_weights, biases: old.biases })
}

fn gaussian_connections(input_size: usize, new_size: usize, output_size: usize, output_biases: Vec<f32>) -> (Connection, Connection) {
    let mut rng = rand::rng();
    let input_weights = gaussian_weights(input_size * new_size, input_size, &mut rng);
    let output_weights = gaussian_weights(new_size * output_size, new_size, &mut rng);
    (Connection { weights: input_weights, biases: vec![0.0; new_size] }, Connection { weights: output_weights, biases: output_biases })
}

fn values_connections(input_size: usize, new_size: usize, output_size: usize, old_biases: Vec<f32>, values: LayerValues) -> Result<(Connection, Connection), String> {
    if values.input_weights.len() != input_size * new_size || values.biases.len() != new_size || values.output_weights.len() != new_size * output_size { return Err("values JSON weights do not match the inserted layer shape".into()); }
    let output_biases = values.output_biases.unwrap_or(old_biases);
    if output_biases.len() != output_size { return Err("values JSON output_biases do not match the next layer".into()); }
    Ok((Connection { weights: values.input_weights, biases: values.biases }, Connection { weights: values.output_weights, biases: output_biases }))
}

fn uniform_weights(count: usize, fan_in: usize, rng: &mut rand::rngs::ThreadRng) -> Vec<f32> {
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..count).map(|_| rng.random_range(-scale..scale)).collect()
}

fn gaussian_weights(count: usize, fan_in: usize, rng: &mut rand::rngs::ThreadRng) -> Vec<f32> {
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..count).map(|_| { let value: f32 = StandardNormal.sample(rng); value * scale }).collect()
}

/// Dense blocks that each read the previous block plus every register written before it.
fn memory_bank_head(mut width: usize, hidden: usize, slots: usize, blocks: usize, outputs: usize, rng: &mut rand::rngs::ThreadRng) -> Vec<Layer> {
    let mut layers = Vec::new();
    let mut bank_total = 0;
    for _ in 0..blocks {
        layers.push(Layer::Dense { inputs: width, outputs: hidden, weights: gaussian_weights(width * hidden, width, rng), biases: vec![0.0; hidden] });
        layers.push(Layer::Relu);
        if slots > 0 {
            layers.push(Layer::MemoryBank { inputs: hidden, slots, weights: gaussian_weights(hidden * slots, hidden, rng), biases: vec![0.0; slots] });
            bank_total += slots;
        }
        width = hidden + bank_total;
    }
    layers.push(Layer::Dense { inputs: width, outputs, weights: gaussian_weights(width * outputs, width, rng), biases: vec![0.0; outputs] });
    layers
}

/// Forward pass for the sequence-shaped layers, which all use `1 x sequence x width` tensors.
fn sequence_forward(layer: &Layer, tensor: &Tensor) -> Result<Tensor, String> {
    match layer {
        Layer::Embedding { vocab, d_model, weights, .. } => {
            let sequence = tensor.values.len();
            for value in &tensor.values {
                let token = *value as usize;
                if token as f32 != *value || token >= *vocab { return Err(format!("token id {value} is outside the vocabulary of {vocab}")); }
            }
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                // The table may be newer on the device than on the host during training, so the
                // rows are always gathered there rather than guessing which copy is current.
                if let Ok(values) = crate::gpu::embedding_gather(&tensor.values, weights, *d_model) {
                    return Tensor::new(1, sequence, *d_model, values);
                }
            }
            let mut values = vec![0.0; sequence * d_model];
            for (position, value) in tensor.values.iter().enumerate() {
                let token = *value as usize;
                values[position * d_model..(position + 1) * d_model].copy_from_slice(&weights[token * d_model..(token + 1) * d_model]);
            }
            Tensor::new(1, sequence, *d_model, values)
        }
        Layer::PositionalEmbedding { max_sequence, d_model, weights, .. } => {
            if tensor.width != *d_model { return Err("positional embedding width does not match the model width".into()); }
            if tensor.height > *max_sequence { return Err(format!("sequence of {} exceeds the maximum of {max_sequence}", tensor.height)); }
            let mut values = tensor.values.clone();
            for position in 0..tensor.height {
                for column in 0..*d_model { values[position * d_model + column] += weights[position * d_model + column]; }
            }
            Tensor::new(1, tensor.height, tensor.width, values)
        }
        Layer::LayerNorm { size, weights, biases } => {
            if tensor.width != *size { return Err("layer norm width does not match the model width".into()); }
            let (values, _, _) = crate::transformer::layer_norm(&tensor.values, tensor.height, *size, weights, biases);
            Tensor::new(1, tensor.height, tensor.width, values)
        }
        Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases } => {
            let shape = crate::transformer::BlockShape { sequence: tensor.height, d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden };
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                if let Ok(values) = crate::gpu::transformer_block_forward(&tensor.values, &shape, weights, biases) {
                    return Tensor::new(1, tensor.height, tensor.width, values);
                }
            }
            let (values, _) = crate::transformer::block_forward(&tensor.values, &shape, weights, biases)?;
            Tensor::new(1, tensor.height, tensor.width, values)
        }
        Layer::TimeDistributedDense { inputs, outputs, weights, biases } => {
            if tensor.width != *inputs { return Err("time-distributed dense width does not match its input".into()); }
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                if let Ok(values) = crate::gpu::matmul_nt(&tensor.values, weights, Some(biases), tensor.height, *outputs, *inputs) {
                    return Tensor::new(1, tensor.height, *outputs, values);
                }
            }
            let mut values = vec![0.0; tensor.height * outputs];
            for row in 0..tensor.height {
                for output in 0..*outputs {
                    let mut sum = biases[output];
                    for column in 0..*inputs { sum += tensor.values[row * inputs + column] * weights[output * inputs + column]; }
                    values[row * outputs + output] = sum;
                }
            }
            Tensor::new(1, tensor.height, *outputs, values)
        }
        _ => Err("unsupported layer in sequence forward pass".into()),
    }
}

/// Backward pass for the sequence-shaped layers.
fn sequence_backward(layer: &Layer, input: &Tensor, gradient: &Tensor) -> Result<(Tensor, GradientData, GradientData), String> {
    match layer {
        Layer::Embedding { vocab, d_model, weights, .. } => {
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                // The table gradient is mostly zeros; scattering it on the device avoids
                // allocating and shipping the whole vocabulary every step.
                if let Ok(buffer) = crate::gpu::embedding_gradient(&input.values, &gradient.values, *vocab, *d_model) {
                    return Ok((Tensor::new(input.channels, input.height, input.width, vec![0.0; input.values.len()])?, GradientData::Device(buffer), Vec::new().into()));
                }
            }
            let mut weight_gradient = vec![0.0; weights.len()];
            for (position, token) in input.values.iter().enumerate() {
                let token = *token as usize;
                for column in 0..*d_model { weight_gradient[token * d_model + column] += gradient.values[position * d_model + column]; }
            }
            // Token ids are not differentiable, so nothing flows further back.
            Ok((Tensor::new(input.channels, input.height, input.width, vec![0.0; input.values.len()])?, weight_gradient.into(), Vec::new().into()))
        }
        Layer::PositionalEmbedding { d_model, weights, .. } => {
            let mut weight_gradient = vec![0.0; weights.len()];
            for position in 0..input.height {
                for column in 0..*d_model { weight_gradient[position * d_model + column] += gradient.values[position * d_model + column]; }
            }
            Ok((gradient.clone(), weight_gradient.into(), Vec::new().into()))
        }
        Layer::LayerNorm { size, weights, biases } => {
            let (_, hat, scales) = crate::transformer::layer_norm(&input.values, input.height, *size, weights, biases);
            let mut gamma_gradient = vec![0.0; *size];
            let mut beta_gradient = vec![0.0; *size];
            let input_gradient = crate::transformer::layer_norm_backward(&gradient.values, &hat, &scales, input.height, *size, weights, &mut gamma_gradient, &mut beta_gradient);
            Ok((Tensor::new(1, input.height, input.width, input_gradient)?, gamma_gradient.into(), beta_gradient.into()))
        }
        Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases } => {
            let shape = crate::transformer::BlockShape { sequence: input.height, d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden };
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                if let Ok((input_gradient, weight_gradient, bias_gradient)) = crate::gpu::transformer_block_backward_device(&input.values, &gradient.values, &shape, weights, biases) {
                    return Ok((Tensor::new(1, input.height, input.width, input_gradient)?, GradientData::Device(weight_gradient), GradientData::Device(bias_gradient)));
                }
            }
            let (_, cache) = crate::transformer::block_forward(&input.values, &shape, weights, biases)?;
            let (input_gradient, weight_gradient, bias_gradient) = crate::transformer::block_backward(&input.values, &gradient.values, &shape, weights, &cache)?;
            Ok((Tensor::new(1, input.height, input.width, input_gradient)?, weight_gradient.into(), bias_gradient.into()))
        }
        Layer::TimeDistributedDense { inputs, outputs, weights, .. } => {
            #[cfg(feature = "gpu")]
            if crate::gpu::enabled() {
                if let Ok((input_gradient, weight_gradient, bias_gradient)) = crate::gpu::time_distributed_backward_device(&gradient.values, &input.values, weights, input.height, *outputs, *inputs) {
                    return Ok((Tensor::new(1, input.height, input.width, input_gradient)?, GradientData::Device(weight_gradient), GradientData::Device(bias_gradient)));
                }
            }
            let mut weight_gradient = vec![0.0; weights.len()];
            let mut bias_gradient = vec![0.0; *outputs];
            let mut input_gradient = vec![0.0; input.values.len()];
            for row in 0..input.height {
                for output in 0..*outputs {
                    let delta = gradient.values[row * outputs + output];
                    bias_gradient[output] += delta;
                    for column in 0..*inputs {
                        weight_gradient[output * inputs + column] += delta * input.values[row * inputs + column];
                        input_gradient[row * inputs + column] += delta * weights[output * inputs + column];
                    }
                }
            }
            Ok((Tensor::new(1, input.height, input.width, input_gradient)?, weight_gradient.into(), bias_gradient.into()))
        }
        _ => Err("unsupported layer in sequence backward pass".into()),
    }
}

/// Projects the current activation into `slots` bank registers that later layers can read.
pub(crate) fn memory_bank_write(input: &Tensor, inputs: usize, slots: usize, weights: &[f32], biases: &[f32]) -> Result<Vec<f32>, String> {
    if input.values.len() != inputs || weights.len() != inputs * slots || biases.len() != slots { return Err("invalid memory bank shape".into()); }
    Ok((0..slots).map(|slot| biases[slot] + input.values.iter().enumerate().map(|(index, value)| value * weights[slot * inputs + index]).sum::<f32>()).collect())
}

/// Splits the incoming gradient into the passthrough activation part and the bank part.
///
/// `bank_before` is the bank length prior to this layer, so this layer owns the accumulator range
/// `bank_before..bank_before + slots`. Because the reverse traversal visits every later reader
/// first, those slots are already fully accumulated by the time this layer is reached.
pub(crate) fn memory_bank_backward(input: &Tensor, inputs: usize, slots: usize, weights: &[f32], bank_before: usize, gradient: &Tensor, bank_gradient: &mut [f32]) -> Result<(Tensor, Vec<f32>, Vec<f32>), String> {
    let bank_after = bank_before + slots;
    if input.values.len() != inputs || weights.len() != inputs * slots || gradient.values.len() != inputs + bank_after || bank_gradient.len() < bank_after { return Err("invalid memory bank backward shape".into()); }
    for (accumulated, incoming) in bank_gradient.iter_mut().zip(&gradient.values[inputs..]) { *accumulated += incoming; }
    let write_gradient = bank_gradient[bank_before..bank_after].to_vec();
    let mut input_gradient = gradient.values[..inputs].to_vec();
    let mut weight_gradient = vec![0.0; weights.len()];
    for slot in 0..slots { for index in 0..inputs {
        weight_gradient[slot * inputs + index] = write_gradient[slot] * input.values[index];
        input_gradient[index] += weights[slot * inputs + index] * write_gradient[slot];
    } }
    Ok((Tensor::new(input.channels, input.height, input.width, input_gradient)?, weight_gradient, write_gradient))
}

fn dense(input: &Tensor, inputs: usize, outputs: usize, weights: &[f32], biases: &[f32]) -> Result<Tensor, String> {
    if input.values.len() != inputs || weights.len() != inputs * outputs || biases.len() != outputs { return Err("invalid dense shape".into()); }
    let values = (0..outputs).map(|output| biases[output] + input.values.iter().enumerate().map(|(index, value)| value * weights[output * inputs + index]).sum::<f32>()).collect();
    Tensor::new(1, 1, outputs, values)
}

fn softmax(mut values: Vec<f32>) -> Vec<f32> { let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max); let total: f32 = values.iter_mut().map(|value| { *value = (*value - max).exp(); *value }).sum(); values.iter_mut().for_each(|value| *value /= total); values }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn convolution_relu_pool_flatten_and_dense_forward() {
        let cnn = Model::with_input_shape(1, 3, 3, vec![Layer::Conv2d { in_channels: 1, out_channels: 1, kernel_size: 2, weights: vec![1.0; 4], biases: vec![0.0] }, Layer::Relu, Layer::MaxPool2d { kernel_size: 2 }, Layer::Flatten, Layer::Dense { inputs: 1, outputs: 2, weights: vec![1.0, -1.0], biases: vec![0.0, 0.0] }]);
        let output = cnn.forward(Tensor::new(1, 3, 3, vec![1.0; 9]).unwrap()).unwrap();
        assert!(output[0] > 0.99 && output[1] < 0.01);
    }

    #[test]
    fn backward_primitives_route_expected_gradients() {
        let input = Tensor::new(1, 2, 2, vec![-1.0, 2.0, 3.0, 1.0]).unwrap();
        let gradient = Tensor::new(1, 2, 2, vec![1.0; 4]).unwrap();
        assert_eq!(relu_backward(&input, &gradient).unwrap().values, vec![0.0, 1.0, 1.0, 1.0]);
        let pooled_gradient = Tensor::new(1, 1, 1, vec![2.0]).unwrap();
        assert_eq!(max_pool2d_backward(&input, 2, &pooled_gradient).unwrap().values, vec![0.0, 0.0, 2.0, 0.0]);
        let (_, weights, biases) = dense_backward(&Tensor::new(1, 1, 2, vec![2.0, 3.0]).unwrap(), 1, &[4.0, 5.0], &Tensor::new(1, 1, 1, vec![2.0]).unwrap()).unwrap();
        assert_eq!(weights, vec![4.0, 6.0]); assert_eq!(biases, vec![2.0]);
    }

    #[test]
    fn batched_spatial_layers_match_individual_samples() {
        let batch = BatchTensor::new(2, 1, 3, 3, (1..=18).map(|value| value as f32).collect()).unwrap();
        let weights = vec![1.0; 4]; let biases = vec![0.0];
        let convolution = batch_conv2d(&batch, 1, 2, &weights, &biases).unwrap();
        assert_eq!(convolution.values, vec![12.0, 16.0, 24.0, 28.0, 48.0, 52.0, 60.0, 64.0]);
        let pooled = batch_max_pool2d(&batch, 3).unwrap();
        assert_eq!(pooled.values, vec![9.0, 18.0]);
    }

    #[test]
    fn reverse_traversal_and_adam_reduce_training_loss() {
        let mut cnn = Model::with_input_shape(1, 2, 2, vec![Layer::Conv2d { in_channels: 1, out_channels: 1, kernel_size: 1, weights: vec![0.5], biases: vec![0.0] }, Layer::Relu, Layer::Flatten, Layer::Dense { inputs: 4, outputs: 2, weights: vec![0.1; 8], biases: vec![0.0; 2] }]);
        let inputs = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let before = cross_entropy(&cnn.forward(Tensor::new(1, 2, 2, inputs[0].clone()).unwrap()).unwrap(), &targets[0]);
        cnn.train(&inputs, &targets, 2, 40, 0.02, LearningFunction::Adam).unwrap();
        let after = cross_entropy(&cnn.forward(Tensor::new(1, 2, 2, inputs[0].clone()).unwrap()).unwrap(), &targets[0]);
        assert!(after < before);
        assert!(cnn.optimizer.as_ref().unwrap().step > 0);
    }

    #[test]
    fn memory_bank_preserves_activations_and_grows_each_block() {
        let cnn = Model::with_input_shape(1, 1, 2, vec![
            Layer::MemoryBank { inputs: 2, slots: 1, weights: vec![1.0, 0.0], biases: vec![0.5] },
            Layer::MemoryBank { inputs: 3, slots: 1, weights: vec![0.0, 0.0, 1.0], biases: vec![0.0] },
        ]);
        let cache = cnn.forward_cached(Tensor::new(1, 1, 2, vec![2.0, -3.0]).unwrap()).unwrap();
        assert_eq!(cache.bank_lengths, vec![0, 1, 2]);
        // First block passes the activation through and appends its own register.
        assert_eq!(cache.activations[1].values, vec![2.0, -3.0, 2.5]);
        // Second block still sees the first register verbatim, then appends its own.
        assert_eq!(cache.activations[2].values, vec![2.0, -3.0, 2.5, 2.5, 2.5]);
    }

    #[test]
    fn memory_bank_gradients_match_finite_differences() {
        let mut cnn = Model::with_input_shape(1, 1, 3, vec![
            Layer::Dense { inputs: 3, outputs: 3, weights: vec![0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.25, 0.35], biases: vec![0.05, -0.05, 0.1] },
            Layer::Relu,
            Layer::MemoryBank { inputs: 3, slots: 2, weights: vec![0.15, -0.25, 0.35, 0.45, 0.1, -0.2], biases: vec![0.02, -0.03] },
            Layer::Dense { inputs: 5, outputs: 3, weights: (0..15).map(|value| value as f32 * 0.03 - 0.2).collect(), biases: vec![0.0; 3] },
            Layer::Relu,
            Layer::MemoryBank { inputs: 3, slots: 2, weights: vec![0.3, -0.15, 0.2, -0.4, 0.05, 0.25], biases: vec![-0.01, 0.04] },
            Layer::Dense { inputs: 7, outputs: 2, weights: (0..14).map(|value| value as f32 * 0.04 - 0.25).collect(), biases: vec![0.0; 2] },
        ]);
        let input = vec![0.5, -0.2, 0.8];
        let target = vec![1.0, 0.0];
        let gradients = cnn.cpu_sample_gradients(&input, &target).unwrap();
        let loss = |cnn: &Model| cross_entropy(&cnn.forward(Tensor::new(1, 1, 3, input.clone()).unwrap()).unwrap(), &target);
        let epsilon = 1e-3;
        fn weights_of(cnn: &mut Model, layer_index: usize) -> &mut Vec<f32> {
            match &mut cnn.layers[layer_index] { Layer::Dense { weights, .. } | Layer::MemoryBank { weights, .. } => weights, _ => unreachable!() }
        }
        // The first bank feeds both later Dense layers, so its gradient only matches if the
        // accumulator collects contributions from every downstream reader.
        for (layer_index, parameter) in [(2usize, 1usize), (2, 4), (5, 0), (0, 3)] {
            let analytic = gradients[layer_index].as_ref().unwrap().weights.to_host()[parameter];
            weights_of(&mut cnn, layer_index)[parameter] += epsilon;
            let high = loss(&cnn);
            weights_of(&mut cnn, layer_index)[parameter] -= 2.0 * epsilon;
            let low = loss(&cnn);
            weights_of(&mut cnn, layer_index)[parameter] += epsilon;
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((analytic - numeric).abs() < 2e-3, "layer {layer_index} weight {parameter}: analytic {analytic} vs numeric {numeric}");
        }
    }

    #[test]
    fn memory_bank_network_trains_and_checkpoints() {
        let mut cnn = Model::with_input_shape(1, 1, 3, vec![
            Layer::Dense { inputs: 3, outputs: 4, weights: (0..12).map(|value| value as f32 * 0.05 - 0.3).collect(), biases: vec![0.0; 4] },
            Layer::Relu,
            Layer::MemoryBank { inputs: 4, slots: 2, weights: (0..8).map(|value| value as f32 * 0.04 - 0.15).collect(), biases: vec![0.0; 2] },
            Layer::Dense { inputs: 6, outputs: 2, weights: (0..12).map(|value| value as f32 * 0.03 - 0.2).collect(), biases: vec![0.0; 2] },
        ]);
        let inputs = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let before = cross_entropy(&cnn.forward(Tensor::new(1, 1, 3, inputs[0].clone()).unwrap()).unwrap(), &targets[0]);
        cnn.train(&inputs, &targets, 2, 60, 0.02, LearningFunction::Adam).unwrap();
        let after = cross_entropy(&cnn.forward(Tensor::new(1, 1, 3, inputs[0].clone()).unwrap()).unwrap(), &targets[0]);
        assert!(after < before, "loss did not improve: {before} -> {after}");
        let path = std::env::temp_dir().join("neuralnet-memory-bank.json");
        save_model(&cnn, &path).unwrap();
        let loaded = load_model(&path).unwrap();
        assert_eq!(loaded.layers.len(), cnn.layers.len());
        assert_eq!(loaded.forward(Tensor::new(1, 1, 3, inputs[0].clone()).unwrap()).unwrap(), cnn.forward(Tensor::new(1, 1, 3, inputs[0].clone()).unwrap()).unwrap());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn convolutional_stem_accepts_a_memory_bank_head() {
        let plain = Model::mnist();
        assert!(!plain.layers.iter().any(|layer| matches!(layer, Layer::MemoryBank { .. })));
        assert!(matches!(plain.layers.last(), Some(Layer::Dense { inputs: 64, outputs: 10, .. })));

        let banked = Model::mnist_convolutional(64, 16, 2).unwrap();
        assert_eq!(banked.layers.iter().filter(|layer| matches!(layer, Layer::MemoryBank { .. })).count(), 2);
        // Conv stem still flattens to 400, then each block widens by the accumulated bank.
        assert!(matches!(banked.layers[7], Layer::Dense { inputs: 400, outputs: 64, .. }));
        assert!(matches!(banked.layers[10], Layer::Dense { inputs: 80, outputs: 64, .. }));
        assert!(matches!(banked.layers.last(), Some(Layer::Dense { inputs: 96, outputs: 10, .. })));

        let prediction = banked.forward(Tensor::new(1, 28, 28, vec![0.3; 784]).unwrap()).unwrap();
        assert_eq!(prediction.len(), 10);
        assert!((prediction.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        // Gradients must reach the bank layers through the convolutional stem.
        let target: Vec<f32> = (0..10).map(|index| if index == 3 { 1.0 } else { 0.0 }).collect();
        let gradients = banked.cpu_sample_gradients(&vec![0.3; 784], &target).unwrap();
        for (index, layer) in banked.layers.iter().enumerate() {
            if matches!(layer, Layer::MemoryBank { .. }) {
                assert!(gradients[index].as_ref().unwrap().weights.to_host().iter().any(|value| value.abs() > 1e-9), "bank at {index} received no gradient");
            }
        }
    }

    /// Straightforward ReLU multilayer perceptron, written independently of the graph engine.
    fn reference_mlp(sizes: &[usize], connections: &[(Vec<f32>, Vec<f32>)], input: &[f32]) -> Vec<f32> {
        let mut values = input.to_vec();
        for (index, (weights, biases)) in connections.iter().enumerate() {
            let inputs = sizes[index];
            let outputs = sizes[index + 1];
            let mut next = vec![0.0; outputs];
            for output in 0..outputs {
                let mut sum = biases[output];
                for unit in 0..inputs { sum += values[unit] * weights[output * inputs + unit]; }
                next[output] = if index + 1 == connections.len() { sum } else { sum.max(0.0) };
            }
            values = next;
        }
        softmax(values)
    }

    /// Writes the pre-migration GGUF layout, which described a model with layer widths alone.
    fn write_legacy_gguf(sizes: &[usize], connections: &[(Vec<f32>, Vec<f32>)]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        gguf::u32put(&mut header, 3);
        gguf::u64put(&mut header, (connections.len() * 2) as u64);
        gguf::u64put(&mut header, 3);
        gguf::meta_str(&mut header, "general.architecture", "neuralnet");
        gguf::meta_u32s(&mut header, "neuralnet.layer_sizes", &sizes.iter().map(|&size| size as u32).collect::<Vec<_>>());
        gguf::meta_u32(&mut header, "general.alignment", gguf::ALIGNMENT as u32);
        let mut offset = 0;
        for (index, (weights, biases)) in connections.iter().enumerate() {
            gguf::tensor(&mut header, &format!("layer.{index}.weight"), &[sizes[index] as u64, sizes[index + 1] as u64], offset);
            offset = gguf::aligned(offset + weights.len() * 4);
            gguf::tensor(&mut header, &format!("layer.{index}.bias"), &[sizes[index + 1] as u64], offset);
            offset = gguf::aligned(offset + biases.len() * 4);
        }
        let mut file = header;
        file.resize(gguf::aligned(file.len()), 0);
        for (weights, biases) in connections {
            gguf::f32s(&mut file, weights);
            file.resize(gguf::aligned(file.len()), 0);
            gguf::f32s(&mut file, biases);
            file.resize(gguf::aligned(file.len()), 0);
        }
        file
    }

    #[test]
    fn legacy_dense_gguf_loads_and_matches_an_independent_reference() {
        let sizes = vec![5, 4, 3, 2];
        let connections: Vec<(Vec<f32>, Vec<f32>)> = sizes.windows(2).enumerate().map(|(index, pair)| {
            let weights = (0..pair[0] * pair[1]).map(|unit| ((unit + index * 3) as f32 * 0.037).sin()).collect();
            let biases = (0..pair[1]).map(|unit| (unit as f32 * 0.11) - 0.2).collect();
            (weights, biases)
        }).collect();

        let path = std::env::temp_dir().join("neuralnet-legacy.gguf");
        fs::write(&path, write_legacy_gguf(&sizes, &connections)).unwrap();
        let model = load_model(&path).unwrap();

        assert_eq!(model.layer_sizes(), sizes);
        assert!(matches!(model.layers.as_slice(), [Layer::Flatten, Layer::Dense { .. }, Layer::Relu, Layer::Dense { .. }, Layer::Relu, Layer::Dense { .. }]));

        for seed in 0..8 {
            let input: Vec<f32> = (0..5).map(|index| ((index * 7 + seed * 13) % 17) as f32 / 17.0 - 0.5).collect();
            let expected = reference_mlp(&sizes, &connections, &input);
            let actual = model.forward(Tensor::new(1, 1, 5, input).unwrap()).unwrap();
            for (expected, actual) in expected.iter().zip(&actual) {
                assert!((expected - actual).abs() < 1e-6, "prediction drift: {expected} vs {actual}");
            }
        }

        // Re-saving migrates the file to the new layout without changing behaviour.
        save_model(&model, &path).unwrap();
        let migrated = load_model(&path).unwrap();
        let input = Tensor::new(1, 1, 5, vec![0.1, -0.2, 0.3, -0.4, 0.5]).unwrap();
        assert_eq!(migrated.forward(input.clone()).unwrap(), model.forward(input).unwrap());
        fs::remove_file(path).unwrap();
    }

    fn deterministic_language_model() -> Model {
        let mut model = Model::language_model(7, 8, 2, 16, 2, 6).unwrap();
        // Replace the random initialisation so the gradient check is reproducible.
        let mut counter = 0u32;
        let mut next = || { counter = counter.wrapping_mul(1664525).wrapping_add(1013904223); ((counter >> 8) as f32 / 8388608.0) - 1.0 };
        for layer in &mut model.layers {
            match layer {
                Layer::Embedding { weights, .. } | Layer::PositionalEmbedding { weights, .. } | Layer::TimeDistributedDense { weights, .. } => { for value in weights.iter_mut() { *value = next() * 0.4; } }
                Layer::TransformerBlock { d_model, weights, biases, .. } => {
                    let gains = weights.len() - 2 * *d_model;
                    for value in weights[..gains].iter_mut() { *value = next() * 0.3; }
                    for value in weights[gains..].iter_mut() { *value = 1.0 + next() * 0.05; }
                    for value in biases.iter_mut() { *value = next() * 0.1; }
                }
                Layer::LayerNorm { weights, biases, .. } => { for value in weights.iter_mut() { *value = 1.0 + next() * 0.05; } for value in biases.iter_mut() { *value = next() * 0.05; } }
                _ => {}
            }
        }
        model
    }

    #[test]
    fn transformer_gradients_match_finite_differences() {
        let mut model = deterministic_language_model();
        let tokens = vec![3u32, 1, 4, 1, 5];
        let targets = vec![1u32, 4, 1, 5, 2];
        let (_, gradients) = model.language_model_step(&tokens, &targets).unwrap();

        let epsilon = 1e-3;
        // Probe every parameter group: embeddings, positions, all four attention projections,
        // both feed-forward matrices, the block layer-norm gains, and the output head.
        let probes: Vec<(usize, &str, Vec<usize>)> = vec![
            (0, "embedding", vec![3 * 8 + 2, 5 * 8 + 7]),
            (1, "positional", vec![0, 2 * 8 + 3]),
            (2, "block0 weights", vec![5, 70, 140, 200, 260, 450, 4 * 64 + 2 * 128, 4 * 64 + 2 * 128 + 8 + 3]),
            (3, "block1 weights", vec![11, 91, 171, 251, 300, 470]),
            (4, "final norm", vec![2]),
            (5, "head", vec![9, 33]),
        ];
        for (index, name, positions) in probes {
            for position in positions {
                let analytic = gradients[index].as_ref().unwrap().weights.to_host()[position];
                let perturb = |model: &mut Model, delta: f32| {
                    match &mut model.layers[index] {
                        Layer::Embedding { weights, .. } | Layer::PositionalEmbedding { weights, .. } | Layer::TransformerBlock { weights, .. } | Layer::LayerNorm { weights, .. } | Layer::TimeDistributedDense { weights, .. } => weights[position] += delta,
                        _ => unreachable!(),
                    }
                };
                perturb(&mut model, epsilon);
                let high = model.language_model_step(&tokens, &targets).unwrap().0;
                perturb(&mut model, -2.0 * epsilon);
                let low = model.language_model_step(&tokens, &targets).unwrap().0;
                perturb(&mut model, epsilon);
                let numeric = (high - low) / (2.0 * epsilon);
                assert!((analytic - numeric).abs() < 2e-3, "{name}[{position}]: analytic {analytic} vs numeric {numeric}");
            }
        }

        // Biases too: feed-forward b1/b2 and both layer-norm shifts live in the bias vector.
        for position in [0usize, 16, 20, 30, 35] {
            let analytic = gradients[2].as_ref().unwrap().biases.to_host()[position];
            let perturb = |model: &mut Model, delta: f32| match &mut model.layers[2] { Layer::TransformerBlock { biases, .. } => biases[position] += delta, _ => unreachable!() };
            perturb(&mut model, epsilon);
            let high = model.language_model_step(&tokens, &targets).unwrap().0;
            perturb(&mut model, -2.0 * epsilon);
            let low = model.language_model_step(&tokens, &targets).unwrap().0;
            perturb(&mut model, epsilon);
            let numeric = (high - low) / (2.0 * epsilon);
            assert!((analytic - numeric).abs() < 2e-3, "block bias[{position}]: analytic {analytic} vs numeric {numeric}");
        }
    }

    #[test]
    fn attention_is_causal() {
        let model = deterministic_language_model();
        let logits = |tokens: &[u32]| {
            let input = Tensor::new(1, 1, tokens.len(), tokens.iter().map(|token| *token as f32).collect()).unwrap();
            model.forward_cached(input).unwrap().activations.pop().unwrap()
        };
        let base = logits(&[3, 1, 4, 1, 5]);
        // Changing the final token must not disturb any earlier position.
        let changed = logits(&[3, 1, 4, 1, 0]);
        for position in 0..4 {
            for column in 0..base.width {
                let left = base.values[position * base.width + column];
                let right = changed.values[position * base.width + column];
                assert!((left - right).abs() < 1e-6, "position {position} changed after editing a later token");
            }
        }
        // Truncating the sequence must leave the surviving prefix identical.
        let prefix = logits(&[3, 1, 4]);
        for position in 0..3 {
            for column in 0..base.width {
                assert!((base.values[position * base.width + column] - prefix.values[position * prefix.width + column]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn transformer_language_model_learns_a_sequence() {
        let mut model = Model::language_model(7, 16, 2, 32, 2, 8).unwrap();
        let tokens = vec![1u32, 2, 3, 4, 5, 6];
        let targets = vec![2u32, 3, 4, 5, 6, 1];
        let sequences = vec![(tokens.clone(), targets.clone()), (tokens.clone(), targets.clone())];
        let interrupted = AtomicBool::new(false);
        let before = model.language_model_step(&tokens, &targets).unwrap().0;
        let after = model.train_language_model_batched(&sequences, 220, 2, 0.01, LearningFunction::AdamW { weight_decay: 0.0 }, &interrupted, |_, _| {}).unwrap();
        assert!(after < before * 0.2, "loss did not fall enough: {before} -> {after}");
        assert_eq!(model.generate(&[1], 5, 8).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_side_adamw_tracks_cpu_and_flushes_its_moments() {
        let base = Model::language_model(48, 64, 4, 128, 1, 16).unwrap();
        let tokens: Vec<u32> = (0..12).map(|value| (value * 5 % 48) as u32).collect();
        let targets: Vec<u32> = (0..12).map(|value| (value * 7 % 48) as u32).collect();
        let sequences = vec![(tokens.clone(), targets.clone())];
        let rule = LearningFunction::AdamW { weight_decay: 0.01 };
        let interrupted = AtomicBool::new(false);

        let mut on_cpu = base.clone();
        let mut on_gpu = base.clone();
        crate::gpu::set_enabled(false);
        on_cpu.train_language_model(&sequences, 10, 0.01, rule, &interrupted, |_, _| {}).unwrap();
        crate::gpu::set_enabled(true);
        on_gpu.train_language_model(&sequences, 10, 0.01, rule, &interrupted, |_, _| {}).unwrap();
        crate::gpu::set_enabled(false);

        // Ten steps of Adam only track if the moments persisted on the device between steps;
        // losing them would restart bias correction every step and change the trajectory.
        let cpu_loss = on_cpu.language_model_step(&tokens, &targets).unwrap().0;
        let gpu_loss = on_gpu.language_model_step(&tokens, &targets).unwrap().0;
        assert!((cpu_loss - gpu_loss).abs() < 5e-3, "device AdamW diverged: CPU {cpu_loss}, GPU {gpu_loss}");

        // Moments live on the device during training and must be copied back before they are read.
        let optimizer = on_gpu.optimizer.as_ref().expect("optimizer state was stored");
        assert_eq!(optimizer.step, 10);
        let moment_energy: f32 = optimizer.second.iter().flatten().map(|value| value.abs()).sum();
        assert!(moment_energy > 0.0, "device moments were never flushed back to the host");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cuda_language_model_evaluation_matches_cpu() {
        let model = Model::language_model(48, 64, 4, 128, 1, 16).unwrap();
        let tokens: Vec<u32> = (0..12).map(|value| (value * 5 % 48) as u32).collect();
        let targets: Vec<u32> = (0..12).map(|value| (value * 7 % 48) as u32).collect();
        crate::gpu::set_enabled(false);
        let cpu = model.language_model_evaluate(&tokens, &targets).unwrap();
        crate::gpu::set_enabled(true);
        let gpu = model.language_model_evaluate(&tokens, &targets).unwrap();
        crate::gpu::set_enabled(false);
        assert!((cpu.0 - gpu.0).abs() < 1e-5, "evaluation loss diverged: CPU {} GPU {}", cpu.0, gpu.0);
        assert_eq!(cpu.1, gpu.1, "evaluation accuracy diverged");
    }

    #[test]
    fn last_row_logits_match_the_full_forward_pass() {
        let model = deterministic_language_model();
        for tokens in [vec![3u32], vec![3, 1, 4], vec![3, 1, 4, 1, 5, 2]] {
            let input = Tensor::new(1, 1, tokens.len(), tokens.iter().map(|token| *token as f32).collect()).unwrap();
            let full = model.forward_cached(input).unwrap().activations.pop().unwrap();
            let expected = &full.values[(full.height - 1) * full.width..];
            let actual = model.last_row_logits(&tokens).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5, "shortcut logits drifted: {actual} vs {expected}");
            }
        }
    }

    #[test]
    fn transformer_model_survives_a_gguf_round_trip() {
        let model = deterministic_language_model();
        let path = std::env::temp_dir().join("neuralnet-transformer.gguf");
        save_model(&model, &path).unwrap();
        let loaded = load_model(&path).unwrap();
        let tokens = vec![3u32, 1, 4];
        let input = Tensor::new(1, 1, 3, tokens.iter().map(|token| *token as f32).collect()).unwrap();
        assert_eq!(loaded.forward_cached(input.clone()).unwrap().activations.pop(), model.forward_cached(input).unwrap().activations.pop());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gguf_round_trip_preserves_every_layer_type() {
        let original = Model::with_input_shape(1, 8, 8, vec![
            Layer::Conv2d { in_channels: 1, out_channels: 2, kernel_size: 3, weights: (0..18).map(|value| value as f32 * 0.1).collect(), biases: vec![0.25, -0.5] },
            Layer::Relu,
            Layer::MaxPool2d { kernel_size: 2 },
            Layer::Flatten,
            Layer::Dense { inputs: 18, outputs: 5, weights: (0..90).map(|value| value as f32 * 0.01).collect(), biases: vec![0.1; 5] },
            Layer::Relu,
            Layer::MemoryBank { inputs: 5, slots: 3, weights: (0..15).map(|value| value as f32 * 0.02).collect(), biases: vec![-0.1; 3] },
            Layer::Dense { inputs: 8, outputs: 2, weights: (0..16).map(|value| value as f32 * 0.03).collect(), biases: vec![0.0; 2] },
        ]);
        let path = std::env::temp_dir().join("neuralnet-graph-roundtrip.gguf");
        save_model(&original, &path).unwrap();
        let loaded = load_model(&path).unwrap();
        assert_eq!(loaded.layers.len(), original.layers.len());
        for (loaded, original) in loaded.layers.iter().zip(&original.layers) {
            assert_eq!(serde_json::to_string(loaded).unwrap(), serde_json::to_string(original).unwrap());
        }
        let input = Tensor::new(1, 8, 8, (0..64).map(|value| value as f32 * 0.05).collect()).unwrap();
        assert_eq!(loaded.forward(input.clone()).unwrap(), original.forward(input).unwrap());
        fs::remove_file(path).unwrap();
    }

    fn predict(model: &Model, input: &[f32]) -> Vec<f32> {
        model.forward(Tensor::new(1, 1, input.len(), input.to_vec()).unwrap()).unwrap()
    }

    fn dense_weights(model: &mut Model, connection: usize) -> (&mut Vec<f32>, &mut Vec<f32>) {
        let position = model.layers.iter().enumerate().filter(|(_, layer)| matches!(layer, Layer::Dense { .. })).map(|(index, _)| index).nth(connection).unwrap();
        match &mut model.layers[position] { Layer::Dense { weights, biases, .. } => (weights, biases), _ => unreachable!() }
    }

    #[test]
    fn same_width_passthrough_preserves_predictions() {
        let mut model = Model::dense(vec![2, 3, 2], 0).unwrap();
        let expected = predict(&model, &[0.2, 0.8]);
        model.insert_layer(1, 3, LayerInit::Passthrough).unwrap();
        assert_eq!(model.layer_sizes(), vec![2, 3, 3, 2]);
        for (expected, actual) in expected.iter().zip(predict(&model, &[0.2, 0.8])) { assert!((expected - actual).abs() < 1e-6); }
    }

    #[test]
    fn adding_passthrough_neurons_preserves_predictions() {
        let mut model = Model::dense(vec![2, 3, 2], 0).unwrap();
        let expected = predict(&model, &[0.2, 0.8]);
        model.add_neurons(1, 1, 2, false).unwrap();
        assert_eq!(model.layer_sizes(), vec![2, 5, 2]);
        for (expected, actual) in expected.iter().zip(predict(&model, &[0.2, 0.8])) { assert!((expected - actual).abs() < 1e-6); }
    }

    #[test]
    fn adding_gaussian_neurons_resizes_connections() {
        let mut model = Model::dense(vec![2, 3, 2], 0).unwrap();
        model.add_neurons(1, 0, 2, true).unwrap();
        assert_eq!(model.layer_sizes(), vec![2, 5, 2]);
        assert_eq!(dense_weights(&mut model, 0).0.len(), 10);
        assert_eq!(dense_weights(&mut model, 1).0.len(), 10);
    }

    #[test]
    fn removing_disconnected_neuron_preserves_predictions() {
        let mut model = Model::dense(vec![2, 3, 2], 0).unwrap();
        { let (weights, biases) = dense_weights(&mut model, 0); weights[2..4].fill(0.0); biases[1] = 0.0; }
        { let (weights, _) = dense_weights(&mut model, 1); for output in 0..2 { weights[output * 3 + 1] = 0.0; } }
        let expected = predict(&model, &[0.2, 0.8]);
        model.remove_neurons(1, &[1]).unwrap();
        assert_eq!(model.layer_sizes(), vec![2, 2, 2]);
        for (expected, actual) in expected.iter().zip(predict(&model, &[0.2, 0.8])) { assert!((expected - actual).abs() < 1e-6); }
    }

    #[test]
    fn removing_hidden_layer_composes_linear_connections() {
        let mut model = Model::dense(vec![2, 2, 2], 0).unwrap();
        { let (weights, biases) = dense_weights(&mut model, 0); weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]); biases.fill(0.0); }
        { let (weights, biases) = dense_weights(&mut model, 1); weights.copy_from_slice(&[2.0, 1.0, 1.0, 2.0]); biases.fill(0.0); }
        let expected = predict(&model, &[0.2, 0.8]);
        model.remove_layer(1).unwrap();
        assert_eq!(model.layer_sizes(), vec![2, 2]);
        for (expected, actual) in expected.iter().zip(predict(&model, &[0.2, 0.8])) { assert!((expected - actual).abs() < 1e-6); }
    }

    #[test]
    fn rejects_invalid_and_unsupported_surgery() {
        let mut model = Model::dense(vec![2, 3, 2], 0).unwrap();
        assert!(model.insert_layer(0, 3, LayerInit::Passthrough).is_err());
        assert!(model.insert_layer(2, 3, LayerInit::Passthrough).is_err());
        assert!(model.add_neurons(0, 0, 1, false).is_err());
        assert!(model.add_neurons(2, 0, 1, false).is_err());
        assert!(model.remove_neurons(1, &[]).is_err());
        assert!(model.remove_neurons(1, &[0, 1, 2]).is_err());
        assert!(model.remove_layer(0).is_err());

        // Surgery must refuse graphs it cannot safely reshape rather than corrupting them.
        let mut banked = Model::mnist_memory_bank(8, 4, 2).unwrap();
        assert!(banked.add_neurons(1, 0, 1, false).is_err());
        let mut convolutional = Model::mnist();
        assert!(convolutional.remove_layer(1).is_err());
    }

    #[test]
    fn cnn_checkpoint_preserves_optimizer_state() {
        let mut cnn = Model::with_input_shape(1, 2, 2, vec![Layer::Flatten, Layer::Dense { inputs: 4, outputs: 2, weights: vec![0.1; 8], biases: vec![0.0; 2] }]);
        cnn.train(&[vec![1.0, 0.0, 0.0, 0.0]], &[vec![1.0, 0.0]], 1, 1, 0.01, LearningFunction::AdamW { weight_decay: 0.01 }).unwrap();
        let path = std::env::temp_dir().join("neuralnet-cnn.json");
        save_model(&cnn, &path).unwrap();
        let loaded = load_model(&path).unwrap();
        assert_eq!(loaded.optimizer.as_ref().unwrap().step, cnn.optimizer.as_ref().unwrap().step);
        fs::remove_file(path).unwrap();
    }
}

fn cross_entropy(prediction: &[f32], target: &[f32]) -> f32 { -prediction.iter().zip(target).map(|(value, target)| target * value.max(1e-7).ln()).sum::<f32>() }