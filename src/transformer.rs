//! Pre-norm transformer block math: layer normalisation, causal multi-head self-attention,
//! and a GELU feed-forward network, with analytic gradients for each.

pub struct BlockShape {
    pub sequence: usize,
    pub d_model: usize,
    pub heads: usize,
    pub ff_hidden: usize,
}

impl BlockShape {
    pub fn head_dim(&self) -> usize { self.d_model / self.heads }
    pub fn weight_count(&self) -> usize { 4 * self.d_model * self.d_model + 2 * self.ff_hidden * self.d_model + 2 * self.d_model }
    pub fn bias_count(&self) -> usize { self.ff_hidden + 3 * self.d_model }

    /// All twelve parameter buffers in the order the CUDA block expects them.
    #[cfg(feature = "gpu")]
    pub(crate) fn parameter_slices<'a>(&self, weights: &'a [f32], biases: &'a [f32]) -> [&'a [f32]; 12] {
        let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma] = self.weight_slices(weights);
        let [b1, b2, ln1_beta, ln2_beta] = self.bias_slices(biases);
        [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma, b1, b2, ln1_beta, ln2_beta]
    }

    /// Offsets into the packed weight vector: wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma.
    fn weight_slices<'a>(&self, weights: &'a [f32]) -> [&'a [f32]; 8] {
        let square = self.d_model * self.d_model;
        let feed = self.ff_hidden * self.d_model;
        let mut at = 0;
        let mut next = |count: usize| { let slice = &weights[at..at + count]; at += count; slice };
        [next(square), next(square), next(square), next(square), next(feed), next(feed), next(self.d_model), next(self.d_model)]
    }

    /// Offsets into the packed bias vector: b1, b2, ln1_beta, ln2_beta.
    fn bias_slices<'a>(&self, biases: &'a [f32]) -> [&'a [f32]; 4] {
        let mut at = 0;
        let mut next = |count: usize| { let slice = &biases[at..at + count]; at += count; slice };
        [next(self.ff_hidden), next(self.d_model), next(self.d_model), next(self.d_model)]
    }
}

pub struct BlockCache {
    normalized_one: Vec<f32>,
    normalized_one_hat: Vec<f32>,
    normalized_one_scale: Vec<f32>,
    queries: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    probabilities: Vec<f32>,
    context: Vec<f32>,
    residual_one: Vec<f32>,
    normalized_two: Vec<f32>,
    normalized_two_hat: Vec<f32>,
    normalized_two_scale: Vec<f32>,
    hidden_pre: Vec<f32>,
    hidden: Vec<f32>,
}

const EPSILON: f32 = 1e-5;

/// Below this many multiply-accumulates the host/device transfer costs more than the kernel saves.
#[cfg(feature = "gpu")]
const CUDA_THRESHOLD: usize = 1 << 20;

pub fn gelu(value: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * value * (1.0 + (COEFFICIENT * (value + 0.044715 * value * value * value)).tanh())
}

pub fn gelu_derivative(value: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_56;
    let inner = COEFFICIENT * (value + 0.044715 * value * value * value);
    let tanh = inner.tanh();
    0.5 * (1.0 + tanh) + 0.5 * value * (1.0 - tanh * tanh) * COEFFICIENT * (1.0 + 3.0 * 0.044715 * value * value)
}

/// Row-wise layer normalisation. Returns the output, the normalised values, and 1/sqrt(var+eps).
pub(crate) fn layer_norm(input: &[f32], rows: usize, width: usize, gamma: &[f32], beta: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut output = vec![0.0; input.len()];
    let mut hat = vec![0.0; input.len()];
    let mut scales = vec![0.0; rows];
    for row in 0..rows {
        let slice = &input[row * width..(row + 1) * width];
        let mean = slice.iter().sum::<f32>() / width as f32;
        let variance = slice.iter().map(|value| (value - mean) * (value - mean)).sum::<f32>() / width as f32;
        let scale = 1.0 / (variance + EPSILON).sqrt();
        scales[row] = scale;
        for column in 0..width {
            let normalized = (slice[column] - mean) * scale;
            hat[row * width + column] = normalized;
            output[row * width + column] = gamma[column] * normalized + beta[column];
        }
    }
    (output, hat, scales)
}

pub(crate) fn layer_norm_backward(gradient: &[f32], hat: &[f32], scales: &[f32], rows: usize, width: usize, gamma: &[f32], gamma_gradient: &mut [f32], beta_gradient: &mut [f32]) -> Vec<f32> {
    let mut input_gradient = vec![0.0; gradient.len()];
    for row in 0..rows {
        let mut sum_dhat = 0.0;
        let mut sum_dhat_hat = 0.0;
        for column in 0..width {
            let index = row * width + column;
            gamma_gradient[column] += gradient[index] * hat[index];
            beta_gradient[column] += gradient[index];
            let dhat = gradient[index] * gamma[column];
            sum_dhat += dhat;
            sum_dhat_hat += dhat * hat[index];
        }
        for column in 0..width {
            let index = row * width + column;
            let dhat = gradient[index] * gamma[column];
            input_gradient[index] = scales[row] / width as f32 * (width as f32 * dhat - sum_dhat - hat[index] * sum_dhat_hat);
        }
    }
    input_gradient
}

/// `output[row][out] = sum_in input[row][in] * weights[out * width_in + in]`.
fn project(input: &[f32], rows: usize, width_in: usize, width_out: usize, weights: &[f32]) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    if crate::gpu::enabled() && rows * width_out * width_in >= CUDA_THRESHOLD {
        if let Ok(result) = crate::gpu::matmul_nt(input, weights, None, rows, width_out, width_in) { return result; }
    }
    project_on_cpu(input, rows, width_in, width_out, weights)
}

pub(crate) fn project_on_cpu(input: &[f32], rows: usize, width_in: usize, width_out: usize, weights: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; rows * width_out];
    for row in 0..rows {
        for out in 0..width_out {
            let mut sum = 0.0;
            for column in 0..width_in { sum += input[row * width_in + column] * weights[out * width_in + column]; }
            output[row * width_out + out] = sum;
        }
    }
    output
}

fn project_backward(input: &[f32], gradient: &[f32], rows: usize, width_in: usize, width_out: usize, weights: &[f32], weight_gradient: &mut [f32], input_gradient: &mut [f32]) {
    #[cfg(feature = "gpu")]
    if crate::gpu::enabled() && rows * width_out * width_in >= CUDA_THRESHOLD {
        if let Ok((left, right)) = crate::gpu::matmul_nt_backward(gradient, input, weights, rows, width_out, width_in) {
            for (total, value) in input_gradient.iter_mut().zip(&left) { *total += value; }
            for (total, value) in weight_gradient.iter_mut().zip(&right) { *total += value; }
            return;
        }
    }
    project_backward_on_cpu(input, gradient, rows, width_in, width_out, weights, weight_gradient, input_gradient)
}

pub(crate) fn project_backward_on_cpu(input: &[f32], gradient: &[f32], rows: usize, width_in: usize, width_out: usize, weights: &[f32], weight_gradient: &mut [f32], input_gradient: &mut [f32]) {
    for row in 0..rows {
        for out in 0..width_out {
            let delta = gradient[row * width_out + out];
            if delta == 0.0 { continue; }
            for column in 0..width_in {
                weight_gradient[out * width_in + column] += delta * input[row * width_in + column];
                input_gradient[row * width_in + column] += delta * weights[out * width_in + column];
            }
        }
    }
}

pub fn block_forward(input: &[f32], shape: &BlockShape, weights: &[f32], biases: &[f32]) -> Result<(Vec<f32>, BlockCache), String> {
    let (sequence, d_model, heads) = (shape.sequence, shape.d_model, shape.heads);
    if d_model == 0 || heads == 0 || d_model % heads != 0 { return Err("transformer d_model must be a positive multiple of heads".into()); }
    if input.len() != sequence * d_model { return Err("transformer input does not match sequence and model width".into()); }
    if weights.len() != shape.weight_count() || biases.len() != shape.bias_count() { return Err("transformer parameter buffers have the wrong size".into()); }    let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma] = shape.weight_slices(weights);
    let [b1, b2, ln1_beta, ln2_beta] = shape.bias_slices(biases);
    let head_dim = shape.head_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let (normalized_one, normalized_one_hat, normalized_one_scale) = layer_norm(input, sequence, d_model, ln1_gamma, ln1_beta);
    let queries = project(&normalized_one, sequence, d_model, d_model, wq);
    let keys = project(&normalized_one, sequence, d_model, d_model, wk);
    let values = project(&normalized_one, sequence, d_model, d_model, wv);

    let mut probabilities = vec![0.0; heads * sequence * sequence];
    let mut context = vec![0.0; sequence * d_model];
    for head in 0..heads {
        let offset = head * head_dim;
        for query in 0..sequence {
            let mut scores = vec![f32::NEG_INFINITY; sequence];
            let mut maximum = f32::NEG_INFINITY;
            for key in 0..=query {
                let mut sum = 0.0;
                for index in 0..head_dim { sum += queries[query * d_model + offset + index] * keys[key * d_model + offset + index]; }
                scores[key] = sum * scale;
                maximum = maximum.max(scores[key]);
            }
            let mut total = 0.0;
            for key in 0..=query { scores[key] = (scores[key] - maximum).exp(); total += scores[key]; }
            for key in 0..=query {
                let probability = scores[key] / total;
                probabilities[(head * sequence + query) * sequence + key] = probability;
                for index in 0..head_dim { context[query * d_model + offset + index] += probability * values[key * d_model + offset + index]; }
            }
        }
    }

    let attention = project(&context, sequence, d_model, d_model, wo);
    let residual_one: Vec<f32> = input.iter().zip(&attention).map(|(left, right)| left + right).collect();
    let (normalized_two, normalized_two_hat, normalized_two_scale) = layer_norm(&residual_one, sequence, d_model, ln2_gamma, ln2_beta);

    let mut hidden_pre = project(&normalized_two, sequence, d_model, shape.ff_hidden, w1);
    for row in 0..sequence { for column in 0..shape.ff_hidden { hidden_pre[row * shape.ff_hidden + column] += b1[column]; } }
    let hidden: Vec<f32> = hidden_pre.iter().map(|value| gelu(*value)).collect();
    let mut feed = project(&hidden, sequence, shape.ff_hidden, d_model, w2);
    for row in 0..sequence { for column in 0..d_model { feed[row * d_model + column] += b2[column]; } }
    let output: Vec<f32> = residual_one.iter().zip(&feed).map(|(left, right)| left + right).collect();

    Ok((output, BlockCache { normalized_one, normalized_one_hat, normalized_one_scale, queries, keys, values, probabilities, context, residual_one, normalized_two, normalized_two_hat, normalized_two_scale, hidden_pre, hidden }))
}

pub fn block_backward(input: &[f32], gradient: &[f32], shape: &BlockShape, weights: &[f32], cache: &BlockCache) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let (sequence, d_model, heads) = (shape.sequence, shape.d_model, shape.heads);
    let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma] = shape.weight_slices(weights);
    let head_dim = shape.head_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut weight_gradient = vec![0.0; shape.weight_count()];
    let mut bias_gradient = vec![0.0; shape.bias_count()];
    {
        let square = d_model * d_model;
        let feed = shape.ff_hidden * d_model;
        let (weight_head, gamma_parts) = weight_gradient.split_at_mut(4 * square + 2 * feed);
        let (wq_g, rest) = weight_head.split_at_mut(square);
        let (wk_g, rest) = rest.split_at_mut(square);
        let (wv_g, rest) = rest.split_at_mut(square);
        let (wo_g, rest) = rest.split_at_mut(square);
        let (w1_g, w2_g) = rest.split_at_mut(feed);
        let (ln1_gamma_g, ln2_gamma_g) = gamma_parts.split_at_mut(d_model);
        let (b1_g, bias_rest) = bias_gradient.split_at_mut(shape.ff_hidden);
        let (b2_g, beta_parts) = bias_rest.split_at_mut(d_model);
        let (ln1_beta_g, ln2_beta_g) = beta_parts.split_at_mut(d_model);

        // Feed-forward branch.
        let mut feed_gradient = gradient.to_vec();
        for row in 0..sequence { for column in 0..d_model { b2_g[column] += feed_gradient[row * d_model + column]; } }
        let mut hidden_gradient = vec![0.0; sequence * shape.ff_hidden];
        project_backward(&cache.hidden, &feed_gradient, sequence, shape.ff_hidden, d_model, w2, w2_g, &mut hidden_gradient);
        for (index, value) in hidden_gradient.iter_mut().enumerate() { *value *= gelu_derivative(cache.hidden_pre[index]); }
        for row in 0..sequence { for column in 0..shape.ff_hidden { b1_g[column] += hidden_gradient[row * shape.ff_hidden + column]; } }
        let mut normalized_two_gradient = vec![0.0; sequence * d_model];
        project_backward(&cache.normalized_two, &hidden_gradient, sequence, d_model, shape.ff_hidden, w1, w1_g, &mut normalized_two_gradient);
        let residual_from_feed = layer_norm_backward(&normalized_two_gradient, &cache.normalized_two_hat, &cache.normalized_two_scale, sequence, d_model, ln2_gamma, ln2_gamma_g, ln2_beta_g);

        // Residual join: the block output is residual_one + feed.
        let mut residual_gradient = vec![0.0; sequence * d_model];
        for index in 0..residual_gradient.len() { residual_gradient[index] = gradient[index] + residual_from_feed[index]; }
        feed_gradient.clear();

        // Attention branch.
        let mut context_gradient = vec![0.0; sequence * d_model];
        project_backward(&cache.context, &residual_gradient, sequence, d_model, d_model, wo, wo_g, &mut context_gradient);

        let mut query_gradient = vec![0.0; sequence * d_model];
        let mut key_gradient = vec![0.0; sequence * d_model];
        let mut value_gradient = vec![0.0; sequence * d_model];
        for head in 0..heads {
            let offset = head * head_dim;
            for query in 0..sequence {
                let mut probability_gradient = vec![0.0; query + 1];
                for key in 0..=query {
                    let probability = cache.probabilities[(head * sequence + query) * sequence + key];
                    let mut sum = 0.0;
                    for index in 0..head_dim {
                        let delta = context_gradient[query * d_model + offset + index];
                        sum += delta * cache.values[key * d_model + offset + index];
                        value_gradient[key * d_model + offset + index] += probability * delta;
                    }
                    probability_gradient[key] = sum;
                }
                let weighted: f32 = (0..=query).map(|key| cache.probabilities[(head * sequence + query) * sequence + key] * probability_gradient[key]).sum();
                for key in 0..=query {
                    let probability = cache.probabilities[(head * sequence + query) * sequence + key];
                    let score_gradient = probability * (probability_gradient[key] - weighted) * scale;
                    for index in 0..head_dim {
                        query_gradient[query * d_model + offset + index] += score_gradient * cache.keys[key * d_model + offset + index];
                        key_gradient[key * d_model + offset + index] += score_gradient * cache.queries[query * d_model + offset + index];
                    }
                }
            }
        }

        let mut normalized_one_gradient = vec![0.0; sequence * d_model];
        project_backward(&cache.normalized_one, &query_gradient, sequence, d_model, d_model, wq, wq_g, &mut normalized_one_gradient);
        project_backward(&cache.normalized_one, &key_gradient, sequence, d_model, d_model, wk, wk_g, &mut normalized_one_gradient);
        project_backward(&cache.normalized_one, &value_gradient, sequence, d_model, d_model, wv, wv_g, &mut normalized_one_gradient);
        let input_from_attention = layer_norm_backward(&normalized_one_gradient, &cache.normalized_one_hat, &cache.normalized_one_scale, sequence, d_model, ln1_gamma, ln1_gamma_g, ln1_beta_g);

        let mut input_gradient = vec![0.0; sequence * d_model];
        for index in 0..input_gradient.len() { input_gradient[index] = residual_gradient[index] + input_from_attention[index]; }
        let _ = (input, &cache.residual_one);
        return Ok((input_gradient, weight_gradient, bias_gradient));
    }
}
