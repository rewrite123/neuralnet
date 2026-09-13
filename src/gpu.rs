use crate::model::{BatchTensor, Tensor};

/// Flat dense connection used by the batched CUDA gradient path.
pub struct DenseLayer { pub weights: Vec<f32>, pub biases: Vec<f32> }

pub struct Gradient { pub weights: Vec<Vec<f32>>, pub biases: Vec<Vec<f32>> }
use cudarc::{driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg}, nvrtc::compile_ptx};
use std::{cell::RefCell, collections::HashMap, env, process::Command, sync::Arc};

const TRAINING_KERNELS: &str = r#"
extern "C" __global__ void forward(const float* x, const float* w, const float* b, float* out, int batch, int input, int output, int relu) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * output) return;
    int sample = index / output; int unit = index % output; float sum = b[unit];
    for (int i = 0; i < input; ++i) sum += x[sample * input + i] * w[unit * input + i];
    out[index] = relu ? fmaxf(sum, 0.0f) : sum;
}
extern "C" __global__ void output_delta(const float* logits, const float* y, float* delta, int batch, int output) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    float maximum = -3.402823466e+38F;
    for (int o = 0; o < output; ++o) maximum = fmaxf(maximum, logits[sample * output + o]);
    float total = 0.0f;
    for (int o = 0; o < output; ++o) total += expf(logits[sample * output + o] - maximum);
    for (int o = 0; o < output; ++o) delta[sample * output + o] = expf(logits[sample * output + o] - maximum) / total - y[sample * output + o];
}
extern "C" __global__ void hidden_delta(const float* h, const float* w, const float* next_delta, float* delta, int batch, int hidden, int next_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * hidden) return;
    int sample = index / hidden; int unit = index % hidden; float sum = 0.0f;
    for (int next = 0; next < next_size; ++next) sum += w[next * hidden + unit] * next_delta[sample * next_size + next];
    delta[index] = h[index] > 0.0f ? sum : 0.0f;
}
extern "C" __global__ void weight_gradient(const float* left, const float* delta, float* gradient, int batch, int left_size, int right_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= left_size * right_size) return;
    int right = index / left_size; int left_index = index % left_size; float sum = 0.0f;
    for (int sample = 0; sample < batch; ++sample) sum += delta[sample * right_size + right] * left[sample * left_size + left_index];
    gradient[index] = sum / batch;
}
extern "C" __global__ void bias_gradient(const float* delta, float* gradient, int batch, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; if (index >= size) return; float sum = 0.0f;
    for (int sample = 0; sample < batch; ++sample) sum += delta[sample * size + index]; gradient[index] = sum / batch;
}
"#;

const CNN_KERNELS: &str = r#"
extern "C" __global__ void conv2d(const float* input, const float* weights, const float* biases, float* output, int channels, int in_height, int in_width, int out_channels, int kernel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height - kernel + 1; int out_width = in_width - kernel + 1;
    if (index >= out_channels * out_height * out_width) return;
    int filter = index / (out_height * out_width); int pixel = index % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; float sum = biases[filter];
    for (int channel = 0; channel < channels; ++channel) for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) sum += input[channel * in_height * in_width + (row + ky) * in_width + column + kx] * weights[((filter * channels + channel) * kernel + ky) * kernel + kx];
    output[index] = sum;
}
extern "C" __global__ void batch_conv2d(const float* input, const float* weights, const float* biases, float* output, int batch, int channels, int in_height, int in_width, int out_channels, int kernel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height - kernel + 1; int out_width = in_width - kernel + 1; int sample_size = out_channels * out_height * out_width;
    if (index >= batch * sample_size) return; int sample = index / sample_size; int within = index % sample_size; int filter = within / (out_height * out_width); int pixel = within % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; float sum = biases[filter];
    for (int channel = 0; channel < channels; ++channel) for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) sum += input[sample * channels * in_height * in_width + channel * in_height * in_width + (row + ky) * in_width + column + kx] * weights[((filter * channels + channel) * kernel + ky) * kernel + kx]; output[index] = sum;
}
extern "C" __global__ void max_pool2d(const float* input, float* output, int channels, int in_height, int in_width, int kernel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height / kernel; int out_width = in_width / kernel;
    if (index >= channels * out_height * out_width) return;
    int channel = index / (out_height * out_width); int pixel = index % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; float maximum = -3.402823466e+38F;
    for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) maximum = fmaxf(maximum, input[channel * in_height * in_width + (row * kernel + ky) * in_width + column * kernel + kx]);
    output[index] = maximum;
}
extern "C" __global__ void batch_max_pool2d(const float* input, float* output, int batch, int channels, int in_height, int in_width, int kernel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height / kernel; int out_width = in_width / kernel; int sample_size = channels * out_height * out_width;
    if (index >= batch * sample_size) return; int sample = index / sample_size; int within = index % sample_size; int channel = within / (out_height * out_width); int pixel = within % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; float maximum = -3.402823466e+38F;
    for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) maximum = fmaxf(maximum, input[sample * channels * in_height * in_width + channel * in_height * in_width + (row * kernel + ky) * in_width + column * kernel + kx]); output[index] = maximum;
}
extern "C" __global__ void relu_backward(const float* input, const float* gradient, float* output, int size) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size) output[i] = input[i] > 0.0f ? gradient[i] : 0.0f; }
extern "C" __global__ void batch_relu_backward(const float* input, const float* gradient, float* output, int size) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size) output[i] = input[i] > 0.0f ? gradient[i] : 0.0f; }
extern "C" __global__ void pool_backward(const float* input, const float* gradient, float* output, int channels, int in_height, int in_width, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height / kernel; int out_width = in_width / kernel;
    if (i >= channels * out_height * out_width) return; int channel = i / (out_height * out_width); int pixel = i % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; int maximum_index = 0; float maximum = -3.402823466e+38F;
    for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) { int at = channel * in_height * in_width + (row * kernel + ky) * in_width + column * kernel + kx; if (input[at] > maximum) { maximum = input[at]; maximum_index = at; } }
    output[maximum_index] = gradient[i];
}
extern "C" __global__ void batch_pool_backward(const float* input, const float* gradient, float* output, int batch, int channels, int in_height, int in_width, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int out_height = in_height / kernel; int out_width = in_width / kernel; int sample_size = channels * out_height * out_width;
    if (i >= batch * sample_size) return; int sample = i / sample_size; int within = i % sample_size; int channel = within / (out_height * out_width); int pixel = within % (out_height * out_width); int row = pixel / out_width; int column = pixel % out_width; int maximum_index = 0; float maximum = -3.402823466e+38F;
    for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) { int at = sample * channels * in_height * in_width + channel * in_height * in_width + (row * kernel + ky) * in_width + column * kernel + kx; if (input[at] > maximum) { maximum = input[at]; maximum_index = at; } } output[maximum_index] = gradient[i];
}
extern "C" __global__ void dense_input_gradient(const float* weights, const float* gradient, float* output, int inputs, int outputs) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= inputs) return; float sum = 0.0f; for (int o = 0; o < outputs; ++o) sum += weights[o * inputs + i] * gradient[o]; output[i] = sum; }
extern "C" __global__ void dense_weight_gradient(const float* input, const float* gradient, float* output, int inputs, int outputs) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= inputs * outputs) return; int output_index = i / inputs; output[i] = gradient[output_index] * input[i % inputs]; }
extern "C" __global__ void conv_input_gradient(const float* gradient, const float* weights, float* output, int channels, int in_height, int in_width, int out_channels, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= channels * in_height * in_width) return; int channel = i / (in_height * in_width); int pixel = i % (in_height * in_width); int row = pixel / in_width; int column = pixel % in_width; int out_height = in_height - kernel + 1; int out_width = in_width - kernel + 1; float sum = 0.0f;
    for (int out = 0; out < out_channels; ++out) for (int ky = 0; ky < kernel; ++ky) for (int kx = 0; kx < kernel; ++kx) { int out_row = row - ky; int out_column = column - kx; if (out_row >= 0 && out_row < out_height && out_column >= 0 && out_column < out_width) sum += gradient[out * out_height * out_width + out_row * out_width + out_column] * weights[((out * channels + channel) * kernel + ky) * kernel + kx]; }
    output[i] = sum;
}
extern "C" __global__ void conv_weight_gradient(const float* input, const float* gradient, float* output, int channels, int in_height, int in_width, int out_channels, int kernel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= out_channels * channels * kernel * kernel) return; int kx = i % kernel; int ky = (i / kernel) % kernel; int channel = (i / (kernel * kernel)) % channels; int out = i / (channels * kernel * kernel); int out_height = in_height - kernel + 1; int out_width = in_width - kernel + 1; float sum = 0.0f;
    for (int row = 0; row < out_height; ++row) for (int column = 0; column < out_width; ++column) sum += gradient[out * out_height * out_width + row * out_width + column] * input[channel * in_height * in_width + (row + ky) * in_width + column + kx]; output[i] = sum;
}
extern "C" __global__ void conv_bias_gradient(const float* gradient, float* output, int out_channels, int out_height, int out_width) { int out = blockIdx.x * blockDim.x + threadIdx.x; if (out >= out_channels) return; float sum = 0.0f; for (int i = 0; i < out_height * out_width; ++i) sum += gradient[out * out_height * out_width + i]; output[out] = sum; }
extern "C" __global__ void adamw_update(float* parameter, const float* gradient, float* first, float* second, int size, float rate, float weight_decay, float beta1_correction, float beta2_correction) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= size) return; float beta1 = 0.9f; float beta2 = 0.999f;
    first[i] = beta1 * first[i] + (1.0f - beta1) * gradient[i]; second[i] = beta2 * second[i] + (1.0f - beta2) * gradient[i] * gradient[i]; parameter[i] *= 1.0f - rate * weight_decay; parameter[i] -= rate * (first[i] / beta1_correction) / (sqrtf(second[i] / beta2_correction) + 1e-8f);
}
// out[row][col] = bias[col] + sum_i left[row][i] * right[col][i]
extern "C" __global__ void matmul_nt(const float* left, const float* right, const float* bias, float* out, int rows, int columns, int inner, int use_bias) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; if (index >= rows * columns) return;
    int row = index / columns; int column = index % columns;
    const float* a = left + (size_t)row * inner; const float* b = right + (size_t)column * inner;
    float sum = use_bias ? bias[column] : 0.0f;
    for (int i = 0; i < inner; ++i) sum += a[i] * b[i];
    out[index] = sum;
}
// left_gradient[row][i] = sum_col out_gradient[row][col] * right[col][i]
extern "C" __global__ void matmul_nt_left_gradient(const float* out_gradient, const float* right, float* left_gradient, int rows, int columns, int inner) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; if (index >= rows * inner) return;
    int row = index / inner; int i = index % inner;
    const float* g = out_gradient + (size_t)row * columns;
    float sum = 0.0f;
    for (int column = 0; column < columns; ++column) sum += g[column] * right[(size_t)column * inner + i];
    left_gradient[index] = sum;
}
// right_gradient[col][i] = sum_row out_gradient[row][col] * left[row][i]
extern "C" __global__ void matmul_nt_right_gradient(const float* out_gradient, const float* left, float* right_gradient, int rows, int columns, int inner) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; if (index >= columns * inner) return;
    int column = index / inner; int i = index % inner;
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) sum += out_gradient[(size_t)row * columns + column] * left[(size_t)row * inner + i];
    right_gradient[index] = sum;
}
extern "C" __global__ void column_sum(const float* values, float* out, int rows, int columns) {
    int column = blockIdx.x * blockDim.x + threadIdx.x; if (column >= columns) return;
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) sum += values[(size_t)row * columns + column];
    out[column] = sum;
}
extern "C" __global__ void softmax_cross_entropy(const float* logits, const float* targets, float* gradient, float* losses, int rows, int columns) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; if (row >= rows) return;
    const float* input = logits + (size_t)row * columns;
    float* output = gradient + (size_t)row * columns;
    float maximum = -3.402823466e+38f;
    for (int column = 0; column < columns; ++column) maximum = fmaxf(maximum, input[column]);
    float total = 0.0f;
    for (int column = 0; column < columns; ++column) total += expf(input[column] - maximum);
    int target = (int)targets[row];
    for (int column = 0; column < columns; ++column) {
        float probability = expf(input[column] - maximum) / total;
        output[column] = (probability - (column == target ? 1.0f : 0.0f)) / rows;
    }
    losses[row] = -(input[target] - maximum - logf(total));
}
extern "C" __global__ void layer_norm_forward(const float* input, const float* gamma, const float* beta, float* out, float* hat, float* scale, int rows, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; if (row >= rows) return;
    const float* x = input + (size_t)row * width;
    float mean = 0.0f; for (int i = 0; i < width; ++i) mean += x[i]; mean /= width;
    float variance = 0.0f; for (int i = 0; i < width; ++i) { float d = x[i] - mean; variance += d * d; } variance /= width;
    float s = rsqrtf(variance + 1e-5f); scale[row] = s;
    for (int i = 0; i < width; ++i) { float h = (x[i] - mean) * s; hat[(size_t)row * width + i] = h; out[(size_t)row * width + i] = gamma[i] * h + beta[i]; }
}
extern "C" __global__ void layer_norm_backward(const float* gradient, const float* hat, const float* scale, const float* gamma, float* input_gradient, float* gamma_gradient, float* beta_gradient, int rows, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; if (row >= rows) return;
    float sum_dhat = 0.0f, sum_dhat_hat = 0.0f;
    for (int i = 0; i < width; ++i) { float dh = gradient[(size_t)row * width + i] * gamma[i]; sum_dhat += dh; sum_dhat_hat += dh * hat[(size_t)row * width + i]; }
    for (int i = 0; i < width; ++i) {
        size_t at = (size_t)row * width + i;
        float dh = gradient[at] * gamma[i];
        input_gradient[at] = scale[row] / width * (width * dh - sum_dhat - hat[at] * sum_dhat_hat);
        atomicAdd(&gamma_gradient[i], gradient[at] * hat[at]);
        atomicAdd(&beta_gradient[i], gradient[at]);
    }
}
extern "C" __global__ void gelu_forward(const float* input, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= size) return;
    float x = input[i]; const float c = 0.7978845608f;
    out[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}
extern "C" __global__ void gelu_backward(const float* pre, const float* gradient, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= size) return;
    float x = pre[i]; const float c = 0.7978845608f;
    float t = tanhf(c * (x + 0.044715f * x * x * x));
    out[i] = gradient[i] * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * 0.044715f * x * x));
}
extern "C" __global__ void add_vectors(const float* left, const float* right, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size) out[i] = left[i] + right[i];
}
extern "C" __global__ void attention_forward(const float* q, const float* k, const float* v, float* probs, float* context, int sequence, int d_model, int head_dim, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = d_model / head_dim; if (index >= heads * sequence) return;
    int head = index / sequence, query = index % sequence, off = head * head_dim;
    float* p = probs + (size_t)(head * sequence + query) * sequence;
    float maximum = -3.402823466e+38f;
    for (int key = 0; key <= query; ++key) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) sum += q[(size_t)query * d_model + off + i] * k[(size_t)key * d_model + off + i];
        p[key] = sum * scale; maximum = fmaxf(maximum, p[key]);
    }
    float total = 0.0f;
    for (int key = 0; key <= query; ++key) { p[key] = expf(p[key] - maximum); total += p[key]; }
    for (int key = 0; key <= query; ++key) p[key] /= total;
    for (int i = 0; i < head_dim; ++i) {
        float sum = 0.0f;
        for (int key = 0; key <= query; ++key) sum += p[key] * v[(size_t)key * d_model + off + i];
        context[(size_t)query * d_model + off + i] = sum;
    }
}
extern "C" __global__ void attention_value_gradient(const float* probs, const float* v, const float* context_gradient, float* probability_gradient, float* v_gradient, int sequence, int d_model, int head_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = d_model / head_dim; if (index >= heads * sequence) return;
    int head = index / sequence, query = index % sequence, off = head * head_dim;
    const float* p = probs + (size_t)(head * sequence + query) * sequence;
    float* g = probability_gradient + (size_t)(head * sequence + query) * sequence;
    const float* cg = context_gradient + (size_t)query * d_model + off;
    for (int key = 0; key <= query; ++key) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            sum += cg[i] * v[(size_t)key * d_model + off + i];
            atomicAdd(&v_gradient[(size_t)key * d_model + off + i], p[key] * cg[i]);
        }
        g[key] = sum;
    }
}
extern "C" __global__ void attention_score_gradient(const float* probs, const float* probability_gradient, const float* q, const float* k, float* q_gradient, float* k_gradient, int sequence, int d_model, int head_dim, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = d_model / head_dim; if (index >= heads * sequence) return;
    int head = index / sequence, query = index % sequence, off = head * head_dim;
    const float* p = probs + (size_t)(head * sequence + query) * sequence;
    const float* g = probability_gradient + (size_t)(head * sequence + query) * sequence;
    float weighted = 0.0f;
    for (int key = 0; key <= query; ++key) weighted += p[key] * g[key];
    for (int key = 0; key <= query; ++key) {
        float ds = p[key] * (g[key] - weighted) * scale;
        for (int i = 0; i < head_dim; ++i) {
            q_gradient[(size_t)query * d_model + off + i] += ds * k[(size_t)key * d_model + off + i];
            atomicAdd(&k_gradient[(size_t)key * d_model + off + i], ds * q[(size_t)query * d_model + off + i]);
        }
    }
}
extern "C" __global__ void embedding_gather(const float* tokens, const float* table, float* out, int sequence, int d_model) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= sequence * d_model) return;
    int position = i / d_model, column = i % d_model;
    int token = (int)tokens[position];
    out[i] = table[(size_t)token * d_model + column];
}
extern "C" __global__ void copy_range(const float* source, float* target, int offset, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < length) target[offset + i] = source[i];
}
extern "C" __global__ void add_into(float* target, const float* other, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size) target[i] += other[i];
}
extern "C" __global__ void scale_into(float* target, float factor, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < size) target[i] *= factor;
}
// Scatters a sequence of row gradients back into a table, so the dense table never leaves the device.
extern "C" __global__ void embedding_gradient(const float* tokens, const float* gradient, float* out, int sequence, int d_model) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= sequence * d_model) return;
    int position = i / d_model, column = i % d_model;
    int token = (int)tokens[position];
    atomicAdd(&out[(size_t)token * d_model + column], gradient[i]);
}
"#;

thread_local! {
    static CNN_RUNTIME: RefCell<Option<(Arc<CudaContext>, Arc<CudaModule>)>> = const { RefCell::new(None) };
}

fn cnn_runtime() -> Result<(Arc<CudaContext>, Arc<CudaModule>), String> {
    ensure_nvrtc_available()?;
    CNN_RUNTIME.with(|runtime| {
        if runtime.borrow().is_none() {
            let context = CudaContext::new(0).map_err(|error| format!("CUDA device: {error}"))?;
            let module = context.load_module(compile_ptx(CNN_KERNELS).map_err(|error| format!("compiling CNN CUDA kernels: {error}"))?).map_err(|error| error.to_string())?;
            *runtime.borrow_mut() = Some((context, module));
        }
        Ok(runtime.borrow().as_ref().unwrap().clone())
    })
}

macro_rules! launch_kernel {
    ($stream:expr, $function:expr, $elements:expr, $($argument:expr),+ $(,)?) => {{
        let mut builder = $stream.launch_builder($function);
        $(builder.arg($argument);)+
        unsafe { builder.launch(LaunchConfig::for_num_elems($elements as u32)) }.map_err(|error| error.to_string())
    }};
}

/// Thread-local so it matches the CUDA runtime and parameter cache, which are also per-thread.
/// A process-global flag would let one test thread divert another's arithmetic onto the GPU.
thread_local! {
    static CUDA_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_enabled(enabled: bool) { CUDA_ENABLED.with(|flag| flag.set(enabled)); }
pub fn enabled() -> bool { CUDA_ENABLED.with(|flag| flag.get()) }

/// Bumped by every write to model parameters, and by every model construction or drop so a
/// recycled heap address can never be served a stale device buffer.
static PARAMETER_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn invalidate_device_cache() { PARAMETER_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }

struct CachedBuffer { epoch: u64, buffer: CudaSlice<f32>, dirty: bool }

thread_local! {
    static PARAMETER_CACHE: RefCell<HashMap<(usize, usize), CachedBuffer>> = RefCell::new(HashMap::new());
}
/// Uploads `parameters` if the cached copy is missing or predates `epoch`, returning its key.
///
/// Only ever call this for model parameters. Activations change without bumping the epoch, so
/// caching them here would serve stale data.
fn ensure_cached(cache: &mut HashMap<(usize, usize), CachedBuffer>, stream: &Arc<CudaStream>, parameters: &[f32], epoch: u64) -> Result<(usize, usize), String> {
    let key = (parameters.as_ptr() as usize, parameters.len());
    // A dirty entry is the authoritative copy, so an epoch bump must not overwrite it from the
    // host. Re-stamp it instead; the host slice it mirrors is still alive because we hold it.
    if let Some(entry) = cache.get_mut(&key) {
        if entry.dirty { entry.epoch = epoch; return Ok(key); }
    }
    if cache.get(&key).is_none_or(|entry| entry.epoch != epoch) {
        cache.retain(|_, entry| entry.epoch == epoch || entry.dirty);
        let buffer = stream.memcpy_stod(parameters).map_err(|error| error.to_string())?;
        cache.insert(key, CachedBuffer { epoch, buffer, dirty: false });
    }
    Ok(key)
}

/// Removes cached buffers that alias a sub-range of `parameters`.
///
/// A packed parameter vector is cached both whole (by the optimizer) and as sub-slices (by the
/// layer that consumes it). Updating the whole buffer in place leaves those sub-slices stale, so
/// they must be dropped and re-uploaded from the host.
fn evict_aliased(cache: &mut HashMap<(usize, usize), CachedBuffer>, parameters: &[f32]) {
    let start = parameters.as_ptr() as usize;
    let end = start + std::mem::size_of_val(parameters);
    cache.retain(|(address, length), _| {
        let entry_end = address + length * std::mem::size_of::<f32>();
        let aliased = *address >= start && entry_end <= end && (*address, *length) != (start, parameters.len());
        !aliased
    });
}

fn with_cached_parameters<R>(stream: &Arc<CudaStream>, parameters: &[f32], action: impl FnOnce(&CudaSlice<f32>) -> Result<R, String>) -> Result<R, String> {
    let epoch = PARAMETER_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    PARAMETER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let key = ensure_cached(&mut cache, stream, parameters, epoch)?;
        action(&cache[&key].buffer)
    })
}

fn with_cached_pair<R>(stream: &Arc<CudaStream>, first: &[f32], second: &[f32], action: impl FnOnce(&CudaSlice<f32>, &CudaSlice<f32>) -> Result<R, String>) -> Result<R, String> {
    with_cached_many(stream, &[first, second], |buffers| action(buffers[0], buffers[1]))
}

fn with_cached_many<R>(stream: &Arc<CudaStream>, parameters: &[&[f32]], action: impl FnOnce(&[&CudaSlice<f32>]) -> Result<R, String>) -> Result<R, String> {
    let epoch = PARAMETER_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    PARAMETER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let mut keys = Vec::with_capacity(parameters.len());
        for slice in parameters { keys.push(ensure_cached(&mut cache, stream, slice, epoch)?); }
        let cache = &*cache;
        let buffers: Vec<&CudaSlice<f32>> = keys.iter().map(|key| &cache[key].buffer).collect();
        action(&buffers)
    })
}

/// Number of parameter tensors currently resident on the device.
pub fn cached_parameter_count() -> usize {
    let epoch = PARAMETER_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    PARAMETER_CACHE.with(|cache| cache.borrow().values().filter(|entry| entry.epoch == epoch).count())
}

/// `out[row][column] = bias[column] + sum_i left[row][i] * right[column][i]`.
pub fn matmul_nt(left: &[f32], right: &[f32], bias: Option<&[f32]>, rows: usize, columns: usize, inner: usize) -> Result<Vec<f32>, String> {
    if left.len() != rows * inner || right.len() != columns * inner { return Err("invalid CUDA matmul shape".into()); }
    if bias.is_some_and(|bias| bias.len() != columns) { return Err("CUDA matmul bias width does not match".into()); }
    let (context, module) = cnn_runtime()?;
    let stream = context.default_stream();
    let function = module.load_function("matmul_nt").map_err(|error| error.to_string())?;
    let device_left = stream.memcpy_stod(left).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
    let use_bias = i32::from(bias.is_some());
    with_cached_pair(&stream, right, bias.unwrap_or(&[0.0]), |device_right, device_bias| {
        let mut output = stream.alloc_zeros::<f32>(rows * columns).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &function, rows * columns, &device_left, device_right, device_bias, &mut output, &rows_i32, &columns_i32, &inner_i32, &use_bias)?;
        stream.memcpy_dtov(&output).map_err(|error| error.to_string())
    })
}

/// Gradients of `matmul_nt` with respect to its left and right operands.
pub fn matmul_nt_backward(out_gradient: &[f32], left: &[f32], right: &[f32], rows: usize, columns: usize, inner: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    if out_gradient.len() != rows * columns || left.len() != rows * inner || right.len() != columns * inner { return Err("invalid CUDA matmul backward shape".into()); }
    let (context, module) = cnn_runtime()?;
    let stream = context.default_stream();
    let left_kernel = module.load_function("matmul_nt_left_gradient").map_err(|error| error.to_string())?;
    let right_kernel = module.load_function("matmul_nt_right_gradient").map_err(|error| error.to_string())?;
    let device_gradient = stream.memcpy_stod(out_gradient).map_err(|error| error.to_string())?;
    let device_left = stream.memcpy_stod(left).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
    with_cached_parameters(&stream, right, |device_right| {
        let mut left_gradient = stream.alloc_zeros::<f32>(rows * inner).map_err(|error| error.to_string())?;
        let mut right_gradient = stream.alloc_zeros::<f32>(columns * inner).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &left_kernel, rows * inner, &device_gradient, device_right, &mut left_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&stream, &right_kernel, columns * inner, &device_gradient, &device_left, &mut right_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        Ok((stream.memcpy_dtov(&left_gradient).map_err(|error| error.to_string())?, stream.memcpy_dtov(&right_gradient).map_err(|error| error.to_string())?))
    })
}

pub fn column_sum(values: &[f32], rows: usize, columns: usize) -> Result<Vec<f32>, String> {
    if values.len() != rows * columns { return Err("invalid CUDA column sum shape".into()); }
    let (context, module) = cnn_runtime()?;
    let stream = context.default_stream();
    let function = module.load_function("column_sum").map_err(|error| error.to_string())?;
    let device_values = stream.memcpy_stod(values).map_err(|error| error.to_string())?;
    let mut output = stream.alloc_zeros::<f32>(columns).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32) = (rows as i32, columns as i32);
    launch_kernel!(&stream, &function, columns, &device_values, &mut output, &rows_i32, &columns_i32)?;
    stream.memcpy_dtov(&output).map_err(|error| error.to_string())
}

/// Computes mean next-token cross-entropy and its logits gradient on the device.
pub fn softmax_cross_entropy(logits: &[f32], targets: &[u32], rows: usize, columns: usize) -> Result<(f32, Vec<f32>), String> {
    if rows == 0 || columns == 0 || logits.len() != rows * columns || targets.len() != rows || targets.iter().any(|target| *target as usize >= columns) {
        return Err("invalid CUDA softmax cross-entropy shape or target".into());
    }
    let (context, module) = cnn_runtime()?;
    let stream = context.default_stream();
    let function = module.load_function("softmax_cross_entropy").map_err(|error| error.to_string())?;
    let device_logits = stream.memcpy_stod(logits).map_err(|error| error.to_string())?;
    let target_values: Vec<f32> = targets.iter().map(|target| *target as f32).collect();
    let device_targets = stream.memcpy_stod(&target_values).map_err(|error| error.to_string())?;
    let mut gradient = stream.alloc_zeros::<f32>(logits.len()).map_err(|error| error.to_string())?;
    let mut losses = stream.alloc_zeros::<f32>(rows).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32) = (rows as i32, columns as i32);
    launch_kernel!(&stream, &function, rows, &device_logits, &device_targets, &mut gradient, &mut losses, &rows_i32, &columns_i32)?;
    let mean_loss = stream.memcpy_dtov(&losses).map_err(|error| error.to_string())?.iter().sum::<f32>() / rows as f32;
    Ok((mean_loss, stream.memcpy_dtov(&gradient).map_err(|error| error.to_string())?))
}

/// Runs the output projection, softmax cross-entropy, and output-head backward pass on CUDA.
///
/// Logits and their vocabulary-sized gradient remain on the device; only the hidden-state
/// gradient needed by the preceding layer is copied back to the host.
pub fn time_distributed_loss_backward_device(input: &[f32], weights: &[f32], biases: &[f32], targets: &[u32], rows: usize, columns: usize, inner: usize) -> Result<(f32, Vec<f32>, DeviceVector, DeviceVector), String> {
    if rows == 0 || columns == 0 || input.len() != rows * inner || weights.len() != columns * inner || biases.len() != columns || targets.len() != rows || targets.iter().any(|target| *target as usize >= columns) {
        return Err("invalid CUDA language-model output shape or target".into());
    }
    let (context, module) = cnn_runtime()?;
    let stream = context.default_stream();
    let matmul = module.load_function("matmul_nt").map_err(|error| error.to_string())?;
    let loss_kernel = module.load_function("softmax_cross_entropy").map_err(|error| error.to_string())?;
    let left_gradient = module.load_function("matmul_nt_left_gradient").map_err(|error| error.to_string())?;
    let right_gradient = module.load_function("matmul_nt_right_gradient").map_err(|error| error.to_string())?;
    let sum = module.load_function("column_sum").map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(input).map_err(|error| error.to_string())?;
    let target_values: Vec<f32> = targets.iter().map(|target| *target as f32).collect();
    let device_targets = stream.memcpy_stod(&target_values).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
    with_cached_pair(&stream, weights, biases, |device_weights, device_biases| {
        let mut logits = stream.alloc_zeros::<f32>(rows * columns).map_err(|error| error.to_string())?;
        let use_bias = 1i32;
        launch_kernel!(&stream, &matmul, rows * columns, &device_input, device_weights, device_biases, &mut logits, &rows_i32, &columns_i32, &inner_i32, &use_bias)?;
        let mut logits_gradient = stream.alloc_zeros::<f32>(rows * columns).map_err(|error| error.to_string())?;
        let mut losses = stream.alloc_zeros::<f32>(rows).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &loss_kernel, rows, &logits, &device_targets, &mut logits_gradient, &mut losses, &rows_i32, &columns_i32)?;
        let mut input_gradient = stream.alloc_zeros::<f32>(rows * inner).map_err(|error| error.to_string())?;
        let mut weight_gradient = stream.alloc_zeros::<f32>(columns * inner).map_err(|error| error.to_string())?;
        let mut bias_gradient = stream.alloc_zeros::<f32>(columns).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &left_gradient, rows * inner, &logits_gradient, device_weights, &mut input_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&stream, &right_gradient, columns * inner, &logits_gradient, &device_input, &mut weight_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&stream, &sum, columns, &logits_gradient, &mut bias_gradient, &rows_i32, &columns_i32)?;
        let mean_loss = stream.memcpy_dtov(&losses).map_err(|error| error.to_string())?.iter().sum::<f32>() / rows as f32;
        Ok((mean_loss, stream.memcpy_dtov(&input_gradient).map_err(|error| error.to_string())?, DeviceVector::new(weight_gradient, columns * inner), DeviceVector::new(bias_gradient, columns)))
    })
}

/// Everything a transformer block needs while it stays on the device.
struct BlockContext<'a> {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    shape: &'a crate::transformer::BlockShape,
}

impl BlockContext<'_> {
    fn function(&self, name: &str) -> Result<cudarc::driver::CudaFunction, String> {
        self.module.load_function(name).map_err(|error| error.to_string())
    }

    fn zeros(&self, count: usize) -> Result<CudaSlice<f32>, String> {
        self.stream.alloc_zeros::<f32>(count).map_err(|error| error.to_string())
    }

    /// `out[row][column] = bias[column] + sum_i left[row][i] * right[column][i]`, device to device.
    fn matmul(&self, left: &CudaSlice<f32>, right: &CudaSlice<f32>, bias: Option<&CudaSlice<f32>>, rows: usize, columns: usize, inner: usize) -> Result<CudaSlice<f32>, String> {
        let function = self.function("matmul_nt")?;
        let mut output = self.zeros(rows * columns)?;
        let placeholder = self.zeros(1)?;
        let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
        let use_bias = i32::from(bias.is_some());
        let bias = bias.unwrap_or(&placeholder);
        launch_kernel!(&self.stream, &function, rows * columns, left, right, bias, &mut output, &rows_i32, &columns_i32, &inner_i32, &use_bias)?;
        Ok(output)
    }

    fn matmul_backward(&self, out_gradient: &CudaSlice<f32>, left: &CudaSlice<f32>, right: &CudaSlice<f32>, rows: usize, columns: usize, inner: usize) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
        let left_kernel = self.function("matmul_nt_left_gradient")?;
        let right_kernel = self.function("matmul_nt_right_gradient")?;
        let mut left_gradient = self.zeros(rows * inner)?;
        let mut right_gradient = self.zeros(columns * inner)?;
        let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
        launch_kernel!(&self.stream, &left_kernel, rows * inner, out_gradient, right, &mut left_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&self.stream, &right_kernel, columns * inner, out_gradient, left, &mut right_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        Ok((left_gradient, right_gradient))
    }

    fn layer_norm(&self, input: &CudaSlice<f32>, gamma: &CudaSlice<f32>, beta: &CudaSlice<f32>) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), String> {
        let function = self.function("layer_norm_forward")?;
        let (rows, width) = (self.shape.sequence, self.shape.d_model);
        let mut output = self.zeros(rows * width)?;
        let mut hat = self.zeros(rows * width)?;
        let mut scale = self.zeros(rows)?;
        let (rows_i32, width_i32) = (rows as i32, width as i32);
        launch_kernel!(&self.stream, &function, rows, input, gamma, beta, &mut output, &mut hat, &mut scale, &rows_i32, &width_i32)?;
        Ok((output, hat, scale))
    }

    fn layer_norm_backward(&self, gradient: &CudaSlice<f32>, hat: &CudaSlice<f32>, scale: &CudaSlice<f32>, gamma: &CudaSlice<f32>) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), String> {
        let function = self.function("layer_norm_backward")?;
        let (rows, width) = (self.shape.sequence, self.shape.d_model);
        let mut input_gradient = self.zeros(rows * width)?;
        let mut gamma_gradient = self.zeros(width)?;
        let mut beta_gradient = self.zeros(width)?;
        let (rows_i32, width_i32) = (rows as i32, width as i32);
        launch_kernel!(&self.stream, &function, rows, gradient, hat, scale, gamma, &mut input_gradient, &mut gamma_gradient, &mut beta_gradient, &rows_i32, &width_i32)?;
        Ok((input_gradient, gamma_gradient, beta_gradient))
    }

    fn add(&self, left: &CudaSlice<f32>, right: &CudaSlice<f32>, count: usize) -> Result<CudaSlice<f32>, String> {
        let function = self.function("add_vectors")?;
        let mut output = self.zeros(count)?;
        let count_i32 = count as i32;
        launch_kernel!(&self.stream, &function, count, left, right, &mut output, &count_i32)?;
        Ok(output)
    }

    fn attention(&self, queries: &CudaSlice<f32>, keys: &CudaSlice<f32>, values: &CudaSlice<f32>) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
        let function = self.function("attention_forward")?;
        let (sequence, d_model, heads) = (self.shape.sequence, self.shape.d_model, self.shape.heads);
        let head_dim = self.shape.head_dim();
        let mut probabilities = self.zeros(heads * sequence * sequence)?;
        let mut context = self.zeros(sequence * d_model)?;
        let (sequence_i32, d_model_i32, head_dim_i32) = (sequence as i32, d_model as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        launch_kernel!(&self.stream, &function, heads * sequence, queries, keys, values, &mut probabilities, &mut context, &sequence_i32, &d_model_i32, &head_dim_i32, &scale)?;
        Ok((probabilities, context))
    }

    fn attention_backward(&self, probabilities: &CudaSlice<f32>, queries: &CudaSlice<f32>, keys: &CudaSlice<f32>, values: &CudaSlice<f32>, context_gradient: &CudaSlice<f32>) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), String> {
        let value_kernel = self.function("attention_value_gradient")?;
        let score_kernel = self.function("attention_score_gradient")?;
        let (sequence, d_model, heads) = (self.shape.sequence, self.shape.d_model, self.shape.heads);
        let head_dim = self.shape.head_dim();
        let mut probability_gradient = self.zeros(heads * sequence * sequence)?;
        let mut value_gradient = self.zeros(sequence * d_model)?;
        let mut query_gradient = self.zeros(sequence * d_model)?;
        let mut key_gradient = self.zeros(sequence * d_model)?;
        let (sequence_i32, d_model_i32, head_dim_i32) = (sequence as i32, d_model as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        launch_kernel!(&self.stream, &value_kernel, heads * sequence, probabilities, values, context_gradient, &mut probability_gradient, &mut value_gradient, &sequence_i32, &d_model_i32, &head_dim_i32)?;
        launch_kernel!(&self.stream, &score_kernel, heads * sequence, probabilities, &probability_gradient, queries, keys, &mut query_gradient, &mut key_gradient, &sequence_i32, &d_model_i32, &head_dim_i32, &scale)?;
        Ok((query_gradient, key_gradient, value_gradient))
    }

    fn gelu(&self, input: &CudaSlice<f32>, count: usize) -> Result<CudaSlice<f32>, String> {
        let function = self.function("gelu_forward")?;
        let mut output = self.zeros(count)?;
        let count_i32 = count as i32;
        launch_kernel!(&self.stream, &function, count, input, &mut output, &count_i32)?;
        Ok(output)
    }

    fn gelu_backward(&self, pre: &CudaSlice<f32>, gradient: &CudaSlice<f32>, count: usize) -> Result<CudaSlice<f32>, String> {
        let function = self.function("gelu_backward")?;
        let mut output = self.zeros(count)?;
        let count_i32 = count as i32;
        launch_kernel!(&self.stream, &function, count, pre, gradient, &mut output, &count_i32)?;
        Ok(output)
    }

    fn column_sum(&self, values: &CudaSlice<f32>, rows: usize, columns: usize) -> Result<CudaSlice<f32>, String> {
        let function = self.function("column_sum")?;
        let mut output = self.zeros(columns)?;
        let (rows_i32, columns_i32) = (rows as i32, columns as i32);
        launch_kernel!(&self.stream, &function, columns, values, &mut output, &rows_i32, &columns_i32)?;
        Ok(output)
    }

    fn download(&self, buffer: &CudaSlice<f32>) -> Result<Vec<f32>, String> {
        self.stream.memcpy_dtov(buffer).map_err(|error| error.to_string())
    }
}

/// Intermediates a block forward pass leaves on the device for its backward pass.
struct BlockActivations {
    normalized_one: CudaSlice<f32>,
    hat_one: CudaSlice<f32>,
    scale_one: CudaSlice<f32>,
    queries: CudaSlice<f32>,
    keys: CudaSlice<f32>,
    values: CudaSlice<f32>,
    probabilities: CudaSlice<f32>,
    context: CudaSlice<f32>,
    residual_one: CudaSlice<f32>,
    normalized_two: CudaSlice<f32>,
    hat_two: CudaSlice<f32>,
    scale_two: CudaSlice<f32>,
    hidden_pre: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    output: CudaSlice<f32>,
}

/// Runs the whole block on the device; only `input` is uploaded.
fn run_block_forward(context: &BlockContext, input: &CudaSlice<f32>, parameters: &[&CudaSlice<f32>]) -> Result<BlockActivations, String> {
    let shape = context.shape;
    let (sequence, d_model, ff_hidden) = (shape.sequence, shape.d_model, shape.ff_hidden);
    let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma, b1, b2, ln1_beta, ln2_beta] = parameters else { return Err("transformer block expects twelve parameter buffers".into()) };

    let (normalized_one, hat_one, scale_one) = context.layer_norm(input, ln1_gamma, ln1_beta)?;
    let queries = context.matmul(&normalized_one, wq, None, sequence, d_model, d_model)?;
    let keys = context.matmul(&normalized_one, wk, None, sequence, d_model, d_model)?;
    let values = context.matmul(&normalized_one, wv, None, sequence, d_model, d_model)?;
    let (probabilities, attention_context) = context.attention(&queries, &keys, &values)?;
    let attention = context.matmul(&attention_context, wo, None, sequence, d_model, d_model)?;
    let residual_one = context.add(input, &attention, sequence * d_model)?;
    let (normalized_two, hat_two, scale_two) = context.layer_norm(&residual_one, ln2_gamma, ln2_beta)?;
    let hidden_pre = context.matmul(&normalized_two, w1, Some(b1), sequence, ff_hidden, d_model)?;
    let hidden = context.gelu(&hidden_pre, sequence * ff_hidden)?;
    let feed = context.matmul(&hidden, w2, Some(b2), sequence, d_model, ff_hidden)?;
    let output = context.add(&residual_one, &feed, sequence * d_model)?;

    Ok(BlockActivations { normalized_one, hat_one, scale_one, queries, keys, values, probabilities, context: attention_context, residual_one, normalized_two, hat_two, scale_two, hidden_pre, hidden, output })
}

pub fn transformer_block_forward(input: &[f32], shape: &crate::transformer::BlockShape, weights: &[f32], biases: &[f32]) -> Result<Vec<f32>, String> {
    let (cuda, module) = cnn_runtime()?;
    let context = BlockContext { stream: cuda.default_stream(), module, shape };
    let device_input = context.stream.memcpy_stod(input).map_err(|error| error.to_string())?;
    let slices = shape.parameter_slices(weights, biases);
    with_cached_many(&context.stream, &slices, |parameters| {
        let activations = run_block_forward(&context, &device_input, parameters)?;
        context.download(&activations.output)
    })
}

pub fn transformer_block_backward(input: &[f32], gradient: &[f32], shape: &crate::transformer::BlockShape, weights: &[f32], biases: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let (cuda, module) = cnn_runtime()?;
    let context = BlockContext { stream: cuda.default_stream(), module, shape };
    let (sequence, d_model, ff_hidden) = (shape.sequence, shape.d_model, shape.ff_hidden);
    let device_input = context.stream.memcpy_stod(input).map_err(|error| error.to_string())?;
    let device_gradient = context.stream.memcpy_stod(gradient).map_err(|error| error.to_string())?;
    let slices = shape.parameter_slices(weights, biases);
    with_cached_many(&context.stream, &slices, |parameters| {
        let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma, ..] = parameters else { return Err("transformer block expects twelve parameter buffers".into()) };
        let saved = run_block_forward(&context, &device_input, parameters)?;

        // Feed-forward branch.
        let bias_two_gradient = context.column_sum(&device_gradient, sequence, d_model)?;
        let (hidden_gradient, w2_gradient) = context.matmul_backward(&device_gradient, &saved.hidden, w2, sequence, d_model, ff_hidden)?;
        let pre_gradient = context.gelu_backward(&saved.hidden_pre, &hidden_gradient, sequence * ff_hidden)?;
        let bias_one_gradient = context.column_sum(&pre_gradient, sequence, ff_hidden)?;
        let (normalized_two_gradient, w1_gradient) = context.matmul_backward(&pre_gradient, &saved.normalized_two, w1, sequence, ff_hidden, d_model)?;
        let (residual_from_feed, ln2_gamma_gradient, ln2_beta_gradient) = context.layer_norm_backward(&normalized_two_gradient, &saved.hat_two, &saved.scale_two, ln2_gamma)?;
        let residual_gradient = context.add(&device_gradient, &residual_from_feed, sequence * d_model)?;

        // Attention branch.
        let (context_gradient, wo_gradient) = context.matmul_backward(&residual_gradient, &saved.context, wo, sequence, d_model, d_model)?;
        let (query_gradient, key_gradient, value_gradient) = context.attention_backward(&saved.probabilities, &saved.queries, &saved.keys, &saved.values, &context_gradient)?;
        let (from_queries, wq_gradient) = context.matmul_backward(&query_gradient, &saved.normalized_one, wq, sequence, d_model, d_model)?;
        let (from_keys, wk_gradient) = context.matmul_backward(&key_gradient, &saved.normalized_one, wk, sequence, d_model, d_model)?;
        let (from_values, wv_gradient) = context.matmul_backward(&value_gradient, &saved.normalized_one, wv, sequence, d_model, d_model)?;
        let projected = context.add(&from_queries, &from_keys, sequence * d_model)?;
        let projected = context.add(&projected, &from_values, sequence * d_model)?;
        let (from_attention, ln1_gamma_gradient, ln1_beta_gradient) = context.layer_norm_backward(&projected, &saved.hat_one, &saved.scale_one, ln1_gamma)?;
        let input_gradient = context.add(&residual_gradient, &from_attention, sequence * d_model)?;

        let mut weight_gradient = Vec::with_capacity(shape.weight_count());
        for buffer in [&wq_gradient, &wk_gradient, &wv_gradient, &wo_gradient, &w1_gradient, &w2_gradient, &ln1_gamma_gradient, &ln2_gamma_gradient] {
            weight_gradient.extend(context.download(buffer)?);
        }
        let mut bias_gradient = Vec::with_capacity(shape.bias_count());
        for buffer in [&bias_one_gradient, &bias_two_gradient, &ln1_beta_gradient, &ln2_beta_gradient] {
            bias_gradient.extend(context.download(buffer)?);
        }
        Ok((context.download(&input_gradient)?, weight_gradient, bias_gradient))
    })
}

/// Concatenates device buffers into one, so a block's packed gradient never reaches the host.
fn concatenate(stream: &Arc<CudaStream>, module: &Arc<CudaModule>, parts: &[&CudaSlice<f32>], lengths: &[usize]) -> Result<DeviceVector, String> {
    let total: usize = lengths.iter().sum();
    let mut output = stream.alloc_zeros::<f32>(total.max(1)).map_err(|error| error.to_string())?;
    let function = module.load_function("copy_range").map_err(|error| error.to_string())?;
    let mut offset = 0usize;
    for (part, length) in parts.iter().zip(lengths) {
        if *length > 0 {
            let (offset_i32, length_i32) = (offset as i32, *length as i32);
            launch_kernel!(stream, &function, *length, *part, &mut output, &offset_i32, &length_i32)?;
        }
        offset += length;
    }
    Ok(DeviceVector::new(output, total))
}

/// Backward pass that leaves the packed weight and bias gradients on the device.
pub fn transformer_block_backward_device(input: &[f32], gradient: &[f32], shape: &crate::transformer::BlockShape, weights: &[f32], biases: &[f32]) -> Result<(Vec<f32>, DeviceVector, DeviceVector), String> {
    let (cuda, module) = cnn_runtime()?;
    let context = BlockContext { stream: cuda.default_stream(), module: module.clone(), shape };
    let (sequence, d_model, ff_hidden) = (shape.sequence, shape.d_model, shape.ff_hidden);
    let device_input = context.stream.memcpy_stod(input).map_err(|error| error.to_string())?;
    let device_gradient = context.stream.memcpy_stod(gradient).map_err(|error| error.to_string())?;
    let slices = shape.parameter_slices(weights, biases);
    with_cached_many(&context.stream, &slices, |parameters| {
        let [wq, wk, wv, wo, w1, w2, ln1_gamma, ln2_gamma, ..] = parameters else { return Err("transformer block expects twelve parameter buffers".into()) };
        let saved = run_block_forward(&context, &device_input, parameters)?;

        let bias_two_gradient = context.column_sum(&device_gradient, sequence, d_model)?;
        let (hidden_gradient, w2_gradient) = context.matmul_backward(&device_gradient, &saved.hidden, w2, sequence, d_model, ff_hidden)?;
        let pre_gradient = context.gelu_backward(&saved.hidden_pre, &hidden_gradient, sequence * ff_hidden)?;
        let bias_one_gradient = context.column_sum(&pre_gradient, sequence, ff_hidden)?;
        let (normalized_two_gradient, w1_gradient) = context.matmul_backward(&pre_gradient, &saved.normalized_two, w1, sequence, ff_hidden, d_model)?;
        let (residual_from_feed, ln2_gamma_gradient, ln2_beta_gradient) = context.layer_norm_backward(&normalized_two_gradient, &saved.hat_two, &saved.scale_two, ln2_gamma)?;
        let residual_gradient = context.add(&device_gradient, &residual_from_feed, sequence * d_model)?;

        let (context_gradient, wo_gradient) = context.matmul_backward(&residual_gradient, &saved.context, wo, sequence, d_model, d_model)?;
        let (query_gradient, key_gradient, value_gradient) = context.attention_backward(&saved.probabilities, &saved.queries, &saved.keys, &saved.values, &context_gradient)?;
        let (from_queries, wq_gradient) = context.matmul_backward(&query_gradient, &saved.normalized_one, wq, sequence, d_model, d_model)?;
        let (from_keys, wk_gradient) = context.matmul_backward(&key_gradient, &saved.normalized_one, wk, sequence, d_model, d_model)?;
        let (from_values, wv_gradient) = context.matmul_backward(&value_gradient, &saved.normalized_one, wv, sequence, d_model, d_model)?;
        let projected = context.add(&from_queries, &from_keys, sequence * d_model)?;
        let projected = context.add(&projected, &from_values, sequence * d_model)?;
        let (from_attention, ln1_gamma_gradient, ln1_beta_gradient) = context.layer_norm_backward(&projected, &saved.hat_one, &saved.scale_one, ln1_gamma)?;
        let input_gradient = context.add(&residual_gradient, &from_attention, sequence * d_model)?;

        let square = d_model * d_model;
        let feed = ff_hidden * d_model;
        let packed_weights = concatenate(&context.stream, &module,
            &[&wq_gradient, &wk_gradient, &wv_gradient, &wo_gradient, &w1_gradient, &w2_gradient, &ln1_gamma_gradient, &ln2_gamma_gradient],
            &[square, square, square, square, feed, feed, d_model, d_model])?;
        let packed_biases = concatenate(&context.stream, &module,
            &[&bias_one_gradient, &bias_two_gradient, &ln1_beta_gradient, &ln2_beta_gradient],
            &[ff_hidden, d_model, d_model, d_model])?;
        Ok((context.download(&input_gradient)?, packed_weights, packed_biases))
    })
}

/// Optimizer moments stay on the device between steps; only `flush_moments` brings them back.
/// They are keyed by host address and length like the parameter cache, but are deliberately not
/// epoch-validated, because the device copy is the authoritative one while training runs.
thread_local! {
    static MOMENT_BUFFERS: RefCell<HashMap<(usize, usize), CudaSlice<f32>>> = RefCell::new(HashMap::new());
}

pub fn clear_moment_buffers() { MOMENT_BUFFERS.with(|cell| cell.borrow_mut().clear()); }

/// Applies one Adam/AdamW step on the device, in place on the cached parameter buffer.
///
/// Parameters are copied back so the host stays authoritative for them; moments are left resident.
pub fn adamw_step(parameters: &mut [f32], gradients: &[f32], first: &mut [f32], second: &mut [f32], learning_rate: f32, weight_decay: f32, step: u32) -> Result<(), String> {
    if parameters.len() != gradients.len() || parameters.len() != first.len() || parameters.len() != second.len() || step == 0 { return Err("invalid CUDA AdamW buffers".into()); }
    if parameters.is_empty() { return Ok(()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("adamw_update").map_err(|error| error.to_string())?;
    let device_gradients = stream.memcpy_stod(gradients).map_err(|error| error.to_string())?;
    let size = parameters.len();
    let size_i32 = size as i32;
    let beta1_correction = 1.0 - 0.9f32.powi(step as i32);
    let beta2_correction = 1.0 - 0.999f32.powi(step as i32);

    let epoch = PARAMETER_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    MOMENT_BUFFERS.with(|moments| {
        let mut moments = moments.borrow_mut();
        for (host, key) in [(&*first, (first.as_ptr() as usize, first.len())), (&*second, (second.as_ptr() as usize, second.len()))] {
            if !moments.contains_key(&key) {
                moments.insert(key, stream.memcpy_stod(host).map_err(|error| error.to_string())?);
            }
        }
        let first_key = (first.as_ptr() as usize, first.len());
        let second_key = (second.as_ptr() as usize, second.len());
        let [device_first, device_second] = moments.get_disjoint_mut([&first_key, &second_key]);
        let (device_first, device_second) = (device_first.ok_or("missing first moment buffer")?, device_second.ok_or("missing second moment buffer")?);

        PARAMETER_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            let key = ensure_cached(&mut cache, &stream, parameters, epoch)?;
            let device_parameters = &mut cache.get_mut(&key).unwrap().buffer;
            launch_kernel!(&stream, &function, size, &mut *device_parameters, &device_gradients, device_first, device_second, &size_i32, &learning_rate, &weight_decay, &beta1_correction, &beta2_correction)?;
            stream.memcpy_dtoh(device_parameters, parameters).map_err(|error| error.to_string())?;
            evict_aliased(&mut cache, parameters);
            Ok(())
        })
    })
}

/// Copies device-resident moments back into their host buffers.
pub fn flush_moments(first: &mut [f32], second: &mut [f32]) -> Result<(), String> {
    let (cuda, _) = cnn_runtime()?;
    let stream = cuda.default_stream();
    MOMENT_BUFFERS.with(|moments| {
        let moments = moments.borrow();
        for (host, key) in [(&mut *first, ()), (&mut *second, ())].into_iter().map(|(host, _)| { let key = (host.as_ptr() as usize, host.len()); (host, key) }) {
            if let Some(buffer) = moments.get(&key) {
                stream.memcpy_dtoh(buffer, host).map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    })
}

/// A gradient buffer that stays in device memory between the backward pass and the optimizer step.
pub struct DeviceVector {
    buffer: CudaSlice<f32>,
    length: usize,
}

impl DeviceVector {
    fn new(buffer: CudaSlice<f32>, length: usize) -> Self { Self { buffer, length } }
    pub fn len(&self) -> usize { self.length }
    pub fn to_host(&self) -> Result<Vec<f32>, String> {
        let (cuda, _) = cnn_runtime()?;
        cuda.default_stream().memcpy_dtov(&self.buffer).map_err(|error| error.to_string())
    }
}

pub fn device_zeros(length: usize) -> Result<DeviceVector, String> {
    let (cuda, _) = cnn_runtime()?;
    let buffer = cuda.default_stream().alloc_zeros::<f32>(length.max(1)).map_err(|error| error.to_string())?;
    Ok(DeviceVector::new(buffer, length))
}

pub fn device_upload(values: &[f32]) -> Result<DeviceVector, String> {
    let (cuda, _) = cnn_runtime()?;
    let padded = if values.is_empty() { vec![0.0f32] } else { values.to_vec() };
    let buffer = cuda.default_stream().memcpy_stod(&padded).map_err(|error| error.to_string())?;
    Ok(DeviceVector::new(buffer, values.len()))
}

pub fn device_add(target: &mut DeviceVector, other: &DeviceVector) -> Result<(), String> {
    if target.length != other.length { return Err("device gradient lengths differ".into()); }
    if target.length == 0 { return Ok(()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("add_into").map_err(|error| error.to_string())?;
    let size = target.length as i32;
    launch_kernel!(&stream, &function, target.length, &mut target.buffer, &other.buffer, &size)?;
    Ok(())
}

pub fn device_scale(target: &mut DeviceVector, factor: f32) -> Result<(), String> {
    if target.length == 0 { return Ok(()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("scale_into").map_err(|error| error.to_string())?;
    let size = target.length as i32;
    launch_kernel!(&stream, &function, target.length, &mut target.buffer, &factor, &size)?;
    Ok(())
}

/// Scatters per-position row gradients into a dense table gradient without leaving the device.
pub fn embedding_gradient(tokens: &[f32], row_gradient: &[f32], rows: usize, d_model: usize) -> Result<DeviceVector, String> {
    if row_gradient.len() != tokens.len() * d_model { return Err("embedding gradient shape does not match the sequence".into()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("embedding_gradient").map_err(|error| error.to_string())?;
    let device_tokens = stream.memcpy_stod(tokens).map_err(|error| error.to_string())?;
    let device_gradient = stream.memcpy_stod(row_gradient).map_err(|error| error.to_string())?;
    let mut output = stream.alloc_zeros::<f32>(rows * d_model).map_err(|error| error.to_string())?;
    let (sequence_i32, d_model_i32) = (tokens.len() as i32, d_model as i32);
    launch_kernel!(&stream, &function, tokens.len() * d_model, &device_tokens, &device_gradient, &mut output, &sequence_i32, &d_model_i32)?;
    Ok(DeviceVector::new(output, rows * d_model))
}

/// Adam/AdamW step consuming a gradient that is already resident on the device.
///
/// Parameters stay on the device; `flush_parameters` brings them back when the host needs them.
pub fn adamw_step_device(parameters: &[f32], gradient: &DeviceVector, first: &mut [f32], second: &mut [f32], learning_rate: f32, weight_decay: f32, step: u32) -> Result<(), String> {
    if parameters.len() != gradient.length || parameters.len() != first.len() || parameters.len() != second.len() || step == 0 { return Err("invalid CUDA AdamW buffers".into()); }
    if parameters.is_empty() { return Ok(()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("adamw_update").map_err(|error| error.to_string())?;
    let size = parameters.len();
    let size_i32 = size as i32;
    let beta1_correction = 1.0 - 0.9f32.powi(step as i32);
    let beta2_correction = 1.0 - 0.999f32.powi(step as i32);
    let epoch = PARAMETER_EPOCH.load(std::sync::atomic::Ordering::Relaxed);

    MOMENT_BUFFERS.with(|moments| {
        let mut moments = moments.borrow_mut();
        let first_key = (first.as_ptr() as usize, first.len());
        let second_key = (second.as_ptr() as usize, second.len());
        for (host, key) in [(&*first, first_key), (&*second, second_key)] {
            if !moments.contains_key(&key) { moments.insert(key, stream.memcpy_stod(host).map_err(|error| error.to_string())?); }
        }
        let [device_first, device_second] = moments.get_disjoint_mut([&first_key, &second_key]);
        let (device_first, device_second) = (device_first.ok_or("missing first moment buffer")?, device_second.ok_or("missing second moment buffer")?);

        PARAMETER_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            let key = ensure_cached(&mut cache, &stream, parameters, epoch)?;
            let entry = cache.get_mut(&key).unwrap();
            entry.dirty = true;
            launch_kernel!(&stream, &function, size, &mut entry.buffer, &gradient.buffer, device_first, device_second, &size_i32, &learning_rate, &weight_decay, &beta1_correction, &beta2_correction)?;
            evict_aliased(&mut cache, parameters);
            Ok(())
        })
    })
}

/// Copies a device-resident parameter buffer back to its host slice.
pub fn flush_parameters(parameters: &mut [f32]) -> Result<(), String> {
    let key = (parameters.as_ptr() as usize, parameters.len());
    let (cuda, _) = cnn_runtime()?;
    let stream = cuda.default_stream();
    PARAMETER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let Some(entry) = cache.get_mut(&key) else { return Ok(()) };
        if !entry.dirty { return Ok(()); }
        stream.memcpy_dtoh(&entry.buffer, parameters).map_err(|error| error.to_string())?;
        entry.dirty = false;
        Ok(())
    })
}


/// Parameter buffers whose device copy is newer than the host copy.
pub fn dirty_parameter_count() -> usize {
    PARAMETER_CACHE.with(|cache| cache.borrow().values().filter(|entry| entry.dirty).count())
}

/// Output-head backward keeping the (vocabulary x width) weight gradient on the device.
pub fn time_distributed_backward_device(out_gradient: &[f32], input: &[f32], weights: &[f32], rows: usize, columns: usize, inner: usize) -> Result<(Vec<f32>, DeviceVector, DeviceVector), String> {
    if out_gradient.len() != rows * columns || input.len() != rows * inner || weights.len() != columns * inner { return Err("invalid CUDA output head backward shape".into()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let left_kernel = module.load_function("matmul_nt_left_gradient").map_err(|error| error.to_string())?;
    let right_kernel = module.load_function("matmul_nt_right_gradient").map_err(|error| error.to_string())?;
    let sum_kernel = module.load_function("column_sum").map_err(|error| error.to_string())?;
    let device_gradient = stream.memcpy_stod(out_gradient).map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(input).map_err(|error| error.to_string())?;
    let (rows_i32, columns_i32, inner_i32) = (rows as i32, columns as i32, inner as i32);
    with_cached_parameters(&stream, weights, |device_weights| {
        let mut input_gradient = stream.alloc_zeros::<f32>(rows * inner).map_err(|error| error.to_string())?;
        let mut weight_gradient = stream.alloc_zeros::<f32>(columns * inner).map_err(|error| error.to_string())?;
        let mut bias_gradient = stream.alloc_zeros::<f32>(columns).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &left_kernel, rows * inner, &device_gradient, device_weights, &mut input_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&stream, &right_kernel, columns * inner, &device_gradient, &device_input, &mut weight_gradient, &rows_i32, &columns_i32, &inner_i32)?;
        launch_kernel!(&stream, &sum_kernel, columns, &device_gradient, &mut bias_gradient, &rows_i32, &columns_i32)?;
        let host_input_gradient = stream.memcpy_dtov(&input_gradient).map_err(|error| error.to_string())?;
        Ok((host_input_gradient, DeviceVector::new(weight_gradient, columns * inner), DeviceVector::new(bias_gradient, columns)))
    })
}

/// Gathers the rows a sequence needs from a device-resident embedding table.
///
/// Only the touched rows come back, so a table whose device copy is newer than the host copy can
/// still be read cheaply.
pub fn embedding_gather(tokens: &[f32], table: &[f32], d_model: usize) -> Result<Vec<f32>, String> {
    if table.len() % d_model != 0 { return Err("embedding table width does not divide its length".into()); }
    let (cuda, module) = cnn_runtime()?;
    let stream = cuda.default_stream();
    let function = module.load_function("embedding_gather").map_err(|error| error.to_string())?;
    let device_tokens = stream.memcpy_stod(tokens).map_err(|error| error.to_string())?;
    let (sequence_i32, d_model_i32) = (tokens.len() as i32, d_model as i32);
    with_cached_parameters(&stream, table, |device_table| {
        let mut output = stream.alloc_zeros::<f32>(tokens.len() * d_model).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &function, tokens.len() * d_model, &device_tokens, device_table, &mut output, &sequence_i32, &d_model_i32)?;
        stream.memcpy_dtov(&output).map_err(|error| error.to_string())
    })
}

pub fn conv2d(input: &Tensor, out_channels: usize, kernel: usize, weights: &[f32], biases: &[f32]) -> Result<Tensor, String> {
    if input.height < kernel || input.width < kernel || weights.len() != out_channels * input.channels * kernel * kernel || biases.len() != out_channels { return Err("invalid CUDA convolution shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream();
    let function = module.load_function("conv2d").map_err(|error| error.to_string())?;
    let height = input.height - kernel + 1; let width = input.width - kernel + 1;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_weights = stream.memcpy_stod(weights).map_err(|error| error.to_string())?; let device_biases = stream.memcpy_stod(biases).map_err(|error| error.to_string())?; let output_len = out_channels * height * width; let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|error| error.to_string())?;
    let channels = input.channels as i32; let in_height = input.height as i32; let in_width = input.width as i32; let out_channels = out_channels as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, output_len, &device_input, &device_weights, &device_biases, &mut output, &channels, &in_height, &in_width, &out_channels, &kernel)?;
    Tensor::new(out_channels as usize, height, width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn max_pool2d(input: &Tensor, kernel: usize) -> Result<Tensor, String> {
    if kernel == 0 || input.height < kernel || input.width < kernel { return Err("invalid CUDA max-pool shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("max_pool2d").map_err(|error| error.to_string())?;
    let height = input.height / kernel; let width = input.width / kernel; let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let output_len = input.channels * height * width; let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|error| error.to_string())?;
    let channels = input.channels as i32; let in_height = input.height as i32; let in_width = input.width as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, output_len, &device_input, &mut output, &channels, &in_height, &in_width, &kernel)?;
    Tensor::new(channels as usize, height, width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn batch_conv2d(input: &BatchTensor, out_channels: usize, kernel: usize, weights: &[f32], biases: &[f32]) -> Result<BatchTensor, String> {
    if input.height < kernel || input.width < kernel || weights.len() != out_channels * input.channels * kernel * kernel || biases.len() != out_channels { return Err("invalid CUDA batch convolution shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("batch_conv2d").map_err(|error| error.to_string())?;
    let height = input.height - kernel + 1; let width = input.width - kernel + 1; let output_len = input.batch * out_channels * height * width;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_weights = stream.memcpy_stod(weights).map_err(|error| error.to_string())?; let device_biases = stream.memcpy_stod(biases).map_err(|error| error.to_string())?; let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|error| error.to_string())?;
    let batch = input.batch as i32; let channels = input.channels as i32; let in_height = input.height as i32; let in_width = input.width as i32; let outputs = out_channels as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, output_len, &device_input, &device_weights, &device_biases, &mut output, &batch, &channels, &in_height, &in_width, &outputs, &kernel)?;
    BatchTensor::new(input.batch, out_channels, height, width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn batch_max_pool2d(input: &BatchTensor, kernel: usize) -> Result<BatchTensor, String> {
    if kernel == 0 || input.height < kernel || input.width < kernel { return Err("invalid CUDA batch max-pool shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("batch_max_pool2d").map_err(|error| error.to_string())?;
    let height = input.height / kernel; let width = input.width / kernel; let output_len = input.batch * input.channels * height * width; let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|error| error.to_string())?;
    let batch = input.batch as i32; let channels = input.channels as i32; let in_height = input.height as i32; let in_width = input.width as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, output_len, &device_input, &mut output, &batch, &channels, &in_height, &in_width, &kernel)?;
    BatchTensor::new(input.batch, input.channels, height, width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn batch_relu_backward(input: &BatchTensor, gradient: &BatchTensor) -> Result<BatchTensor, String> {
    if input.batch != gradient.batch || input.channels != gradient.channels || input.height != gradient.height || input.width != gradient.width { return Err("invalid CUDA batched ReLU backward shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("batch_relu_backward").map_err(|error| error.to_string())?; let size = input.values.len(); let size_i32 = size as i32; let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let mut output = stream.alloc_zeros::<f32>(size).map_err(|error| error.to_string())?;
    launch_kernel!(&stream, &function, size, &device_input, &device_gradient, &mut output, &size_i32)?;
    BatchTensor::new(input.batch, input.channels, input.height, input.width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn batch_max_pool2d_backward(input: &BatchTensor, kernel: usize, gradient: &BatchTensor) -> Result<BatchTensor, String> {
    if kernel == 0 || input.height < kernel || input.width < kernel || gradient.batch != input.batch || gradient.channels != input.channels || gradient.height != input.height / kernel || gradient.width != input.width / kernel { return Err("invalid CUDA batched max-pool backward shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("batch_pool_backward").map_err(|error| error.to_string())?; let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let mut output = stream.alloc_zeros::<f32>(input.values.len()).map_err(|error| error.to_string())?; let batch = input.batch as i32; let channels = input.channels as i32; let height = input.height as i32; let width = input.width as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, gradient.values.len(), &device_input, &device_gradient, &mut output, &batch, &channels, &height, &width, &kernel)?;
    BatchTensor::new(input.batch, input.channels, input.height, input.width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn adamw_update(parameters: &mut [f32], gradients: &[f32], first: &mut [f32], second: &mut [f32], learning_rate: f32, weight_decay: f32, step: u32) -> Result<(), String> {
    if parameters.len() != gradients.len() || parameters.len() != first.len() || parameters.len() != second.len() || step == 0 { return Err("invalid CUDA AdamW update buffers".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("adamw_update").map_err(|error| error.to_string())?;
    let mut device_parameters = stream.memcpy_stod(parameters).map_err(|error| error.to_string())?; let device_gradients = stream.memcpy_stod(gradients).map_err(|error| error.to_string())?; let mut device_first = stream.memcpy_stod(first).map_err(|error| error.to_string())?; let mut device_second = stream.memcpy_stod(second).map_err(|error| error.to_string())?;
    let size = parameters.len(); let size_i32 = size as i32; let beta1_correction = 1.0 - 0.9_f32.powi(step as i32); let beta2_correction = 1.0 - 0.999_f32.powi(step as i32);
    launch_kernel!(&stream, &function, size, &mut device_parameters, &device_gradients, &mut device_first, &mut device_second, &size_i32, &learning_rate, &weight_decay, &beta1_correction, &beta2_correction)?;
    parameters.copy_from_slice(&stream.memcpy_dtov(&device_parameters).map_err(|error| error.to_string())?); first.copy_from_slice(&stream.memcpy_dtov(&device_first).map_err(|error| error.to_string())?); second.copy_from_slice(&stream.memcpy_dtov(&device_second).map_err(|error| error.to_string())?);
    Ok(())
}

pub fn cnn_relu_backward(input: &Tensor, gradient: &Tensor) -> Result<Tensor, String> {
    if input.values.len() != gradient.values.len() { return Err("CUDA ReLU gradient shape does not match input".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("relu_backward").map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let length = input.values.len(); let mut output = stream.alloc_zeros::<f32>(length).map_err(|error| error.to_string())?; let length_i32 = length as i32;
    launch_kernel!(&stream, &function, length, &device_input, &device_gradient, &mut output, &length_i32)?;
    Tensor::new(input.channels, input.height, input.width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn cnn_max_pool2d_backward(input: &Tensor, kernel: usize, gradient: &Tensor) -> Result<Tensor, String> {
    if kernel == 0 || input.height < kernel || input.width < kernel || gradient.values.len() != input.channels * (input.height / kernel) * (input.width / kernel) { return Err("invalid CUDA max-pool backward shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let function = module.load_function("pool_backward").map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let mut output = stream.alloc_zeros::<f32>(input.values.len()).map_err(|error| error.to_string())?; let channels = input.channels as i32; let height = input.height as i32; let width = input.width as i32; let kernel = kernel as i32;
    launch_kernel!(&stream, &function, gradient.values.len(), &device_input, &device_gradient, &mut output, &channels, &height, &width, &kernel)?;
    Tensor::new(input.channels, input.height, input.width, stream.memcpy_dtov(&output).map_err(|error| error.to_string())?)
}

pub fn cnn_dense_backward(input: &Tensor, outputs: usize, weights: &[f32], gradient: &Tensor) -> Result<(Tensor, Vec<f32>, Vec<f32>), String> {
    let inputs = input.values.len(); if weights.len() != inputs * outputs || gradient.values.len() != outputs { return Err("invalid CUDA dense backward shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let input_kernel = module.load_function("dense_input_gradient").map_err(|error| error.to_string())?; let weight_kernel = module.load_function("dense_weight_gradient").map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_weights = stream.memcpy_stod(weights).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let mut input_result = stream.alloc_zeros::<f32>(inputs).map_err(|error| error.to_string())?; let mut weight_result = stream.alloc_zeros::<f32>(weights.len()).map_err(|error| error.to_string())?; let inputs_i32 = inputs as i32; let outputs_i32 = outputs as i32;
    launch_kernel!(&stream, &input_kernel, inputs, &device_weights, &device_gradient, &mut input_result, &inputs_i32, &outputs_i32)?; launch_kernel!(&stream, &weight_kernel, weights.len(), &device_input, &device_gradient, &mut weight_result, &inputs_i32, &outputs_i32)?;
    Ok((Tensor::new(input.channels, input.height, input.width, stream.memcpy_dtov(&input_result).map_err(|error| error.to_string())?)?, stream.memcpy_dtov(&weight_result).map_err(|error| error.to_string())?, gradient.values.clone()))
}

pub fn cnn_conv2d_backward(input: &Tensor, out_channels: usize, kernel: usize, weights: &[f32], gradient: &Tensor) -> Result<(Tensor, Vec<f32>, Vec<f32>), String> {
    let out_height = input.height.checked_sub(kernel).ok_or("invalid CUDA convolution kernel")? + 1; let out_width = input.width.checked_sub(kernel).ok_or("invalid CUDA convolution kernel")? + 1;
    if weights.len() != out_channels * input.channels * kernel * kernel || gradient.values.len() != out_channels * out_height * out_width { return Err("invalid CUDA convolution backward shape".into()); }
    let (context, module) = cnn_runtime()?; let stream = context.default_stream(); let input_kernel = module.load_function("conv_input_gradient").map_err(|error| error.to_string())?; let weight_kernel = module.load_function("conv_weight_gradient").map_err(|error| error.to_string())?; let bias_kernel = module.load_function("conv_bias_gradient").map_err(|error| error.to_string())?;
    let device_input = stream.memcpy_stod(&input.values).map_err(|error| error.to_string())?; let device_weights = stream.memcpy_stod(weights).map_err(|error| error.to_string())?; let device_gradient = stream.memcpy_stod(&gradient.values).map_err(|error| error.to_string())?; let mut input_result = stream.alloc_zeros::<f32>(input.values.len()).map_err(|error| error.to_string())?; let mut weight_result = stream.alloc_zeros::<f32>(weights.len()).map_err(|error| error.to_string())?; let mut bias_result = stream.alloc_zeros::<f32>(out_channels).map_err(|error| error.to_string())?; let channels = input.channels as i32; let height = input.height as i32; let width = input.width as i32; let outputs = out_channels as i32; let kernel_i32 = kernel as i32; let out_height_i32 = out_height as i32; let out_width_i32 = out_width as i32;
    launch_kernel!(&stream, &input_kernel, input.values.len(), &device_gradient, &device_weights, &mut input_result, &channels, &height, &width, &outputs, &kernel_i32)?; launch_kernel!(&stream, &weight_kernel, weights.len(), &device_input, &device_gradient, &mut weight_result, &channels, &height, &width, &outputs, &kernel_i32)?; launch_kernel!(&stream, &bias_kernel, out_channels, &device_gradient, &mut bias_result, &outputs, &out_height_i32, &out_width_i32)?;
    Ok((Tensor::new(input.channels, input.height, input.width, stream.memcpy_dtov(&input_result).map_err(|error| error.to_string())?)?, stream.memcpy_dtov(&weight_result).map_err(|error| error.to_string())?, stream.memcpy_dtov(&bias_result).map_err(|error| error.to_string())?))
}

pub fn batch_gradient(sizes: &[usize], layers: &[DenseLayer], inputs: &[Vec<f32>], outputs: &[Vec<f32>]) -> Result<Gradient, String> {
    ensure_nvrtc_available()?;
    if sizes.len() < 2 || layers.len() + 1 != sizes.len() { return Err("invalid dense network layout".into()); }
    if inputs.is_empty() || inputs.len() != outputs.len() { return Err("CUDA batch inputs and outputs must have equal non-zero lengths".into()); }
    let batch = inputs.len();
    if inputs.iter().any(|row| row.len() != sizes[0]) || outputs.iter().any(|row| row.len() != *sizes.last().unwrap()) { return Err("CUDA batch shape does not match model".into()); }

    let ctx = CudaContext::new(0).map_err(|error| format!("CUDA device: {error}"))?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(compile_ptx(TRAINING_KERNELS).map_err(|error| format!("compiling CUDA kernels: {error}"))?).map_err(|error| error.to_string())?;
    let forward = module.load_function("forward").map_err(|error| error.to_string())?;
    let output_delta = module.load_function("output_delta").map_err(|error| error.to_string())?;
    let hidden_delta = module.load_function("hidden_delta").map_err(|error| error.to_string())?;
    let weight_gradient = module.load_function("weight_gradient").map_err(|error| error.to_string())?;
    let bias_gradient = module.load_function("bias_gradient").map_err(|error| error.to_string())?;
    let batch_i32 = batch as i32;

    let flat_inputs: Vec<f32> = inputs.iter().flatten().copied().collect();
    let flat_outputs: Vec<f32> = outputs.iter().flatten().copied().collect();
    let mut activations = vec![stream.memcpy_stod(&flat_inputs).map_err(|error| error.to_string())?];
    let weights: Vec<_> = layers.iter().map(|layer| stream.memcpy_stod(&layer.weights).map_err(|error| error.to_string())).collect::<Result<_, _>>()?;
    let biases: Vec<_> = layers.iter().map(|layer| stream.memcpy_stod(&layer.biases).map_err(|error| error.to_string())).collect::<Result<_, _>>()?;

    for index in 0..layers.len() {
        let input_i32 = sizes[index] as i32;
        let output_i32 = sizes[index + 1] as i32;
        let relu = (index + 1 != layers.len()) as i32;
        let mut output = stream.alloc_zeros::<f32>(batch * sizes[index + 1]).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &forward, batch * sizes[index + 1], &activations[index], &weights[index], &biases[index], &mut output, &batch_i32, &input_i32, &output_i32, &relu)?;
        activations.push(output);
    }

    let targets = stream.memcpy_stod(&flat_outputs).map_err(|error| error.to_string())?;
    let last = layers.len() - 1;
    let output_i32 = sizes[last + 1] as i32;
    let mut last_delta = stream.alloc_zeros::<f32>(batch * sizes[last + 1]).map_err(|error| error.to_string())?;
    launch_kernel!(&stream, &output_delta, batch, &activations[last + 1], &targets, &mut last_delta, &batch_i32, &output_i32)?;
    let mut deltas: Vec<CudaSlice<f32>> = (0..layers.len() - 1).map(|index| stream.alloc_zeros::<f32>(batch * sizes[index + 1]).map_err(|error| error.to_string())).collect::<Result<_, _>>()?;
    deltas.push(last_delta);

    for index in (0..last).rev() {
        let hidden_i32 = sizes[index + 1] as i32;
        let next_i32 = sizes[index + 2] as i32;
        let (before, after) = deltas.split_at_mut(index + 1);
        launch_kernel!(&stream, &hidden_delta, batch * sizes[index + 1], &activations[index + 1], &weights[index + 1], &after[0], &mut before[index], &batch_i32, &hidden_i32, &next_i32)?;
    }

    let mut weight_gradients = Vec::with_capacity(layers.len());
    let mut bias_gradients = Vec::with_capacity(layers.len());
    for index in 0..layers.len() {
        let input_i32 = sizes[index] as i32;
        let output_i32 = sizes[index + 1] as i32;
        let mut weight_gradient_buffer = stream.alloc_zeros::<f32>(sizes[index] * sizes[index + 1]).map_err(|error| error.to_string())?;
        let mut bias_gradient_buffer = stream.alloc_zeros::<f32>(sizes[index + 1]).map_err(|error| error.to_string())?;
        launch_kernel!(&stream, &weight_gradient, sizes[index] * sizes[index + 1], &activations[index], &deltas[index], &mut weight_gradient_buffer, &batch_i32, &input_i32, &output_i32)?;
        launch_kernel!(&stream, &bias_gradient, sizes[index + 1], &deltas[index], &mut bias_gradient_buffer, &batch_i32, &output_i32)?;
        weight_gradients.push(stream.memcpy_dtov(&weight_gradient_buffer).map_err(|error| error.to_string())?);
        bias_gradients.push(stream.memcpy_dtov(&bias_gradient_buffer).map_err(|error| error.to_string())?);
    }
    Ok(Gradient { weights: weight_gradients, biases: bias_gradients })
}

fn ensure_nvrtc_available() -> Result<(), String> {
    let output = Command::new("ldconfig").arg("-p").output().map_err(|error| format!("checking CUDA runtime libraries: {error}"))?;
    let in_linker_cache = String::from_utf8_lossy(&output.stdout).contains("libnvrtc.so");
    let in_library_path = env::var_os("LD_LIBRARY_PATH").is_some_and(|paths| env::split_paths(&paths).any(|path| path.join("libnvrtc.so").is_file()));
    if in_linker_cache || in_library_path { Ok(()) } else { Err("CUDA training requires libnvrtc.so from the CUDA Toolkit. Install a CUDA Toolkit version compatible with the NVIDIA driver, then make its lib64 directory discoverable through LD_LIBRARY_PATH or ldconfig.".into()) }
}

#[cfg(test)]
mod tests {
    use crate::{model::{batch_conv2d as cpu_batch_conv2d, batch_max_pool2d as cpu_batch_max_pool2d, batch_max_pool2d_backward as cpu_batch_pool_backward, batch_relu_backward as cpu_batch_relu_backward, conv2d as cpu_conv2d, max_pool2d as cpu_max_pool2d, BatchTensor, Model, Layer as CnnLayer, Tensor}, learn_functions::LearningFunction};

    #[test]
    fn batch_gradient_matches_cpu_reference_for_deep_network() {
        let sizes = vec![3, 4, 5, 2];
        let model = Model::dense(sizes.clone(), 0).unwrap();
        let inputs = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6], vec![0.7, 0.8, 0.9]];
        let outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let (chain_sizes, dense) = model.dense_cuda_chain().expect("plain dense stack");
        assert_eq!(chain_sizes, sizes);
        let gpu = super::batch_gradient(&sizes, &dense, &inputs, &outputs).unwrap();

        // Independent reference: average the graph engine's per-sample CPU gradients.
        let mut weights: Vec<Vec<f32>> = dense.iter().map(|layer| vec![0.0; layer.weights.len()]).collect();
        let mut biases: Vec<Vec<f32>> = dense.iter().map(|layer| vec![0.0; layer.biases.len()]).collect();
        for (input, target) in inputs.iter().zip(&outputs) {
            for (connection, gradient) in model.cpu_sample_gradients(input, target).unwrap().into_iter().flatten().enumerate() {
                for (total, value) in weights[connection].iter_mut().zip(&gradient.weights.to_host()) { *total += value / inputs.len() as f32; }
                for (total, value) in biases[connection].iter_mut().zip(&gradient.biases.to_host()) { *total += value / inputs.len() as f32; }
            }
        }
        for (cpu_layer, gpu_layer) in weights.iter().zip(&gpu.weights).chain(biases.iter().zip(&gpu.biases)) {
            for (cpu_value, gpu_value) in cpu_layer.iter().zip(gpu_layer) { assert!((cpu_value - gpu_value).abs() < 1e-5, "CPU {cpu_value}, GPU {gpu_value}"); }
        }
    }

    #[test]
    fn matmul_and_its_gradients_match_cpu_reference() {
        let (rows, columns, inner) = (7usize, 5usize, 9usize);
        let left: Vec<f32> = (0..rows * inner).map(|value| ((value * 13 % 31) as f32 * 0.07) - 1.0).collect();
        let right: Vec<f32> = (0..columns * inner).map(|value| ((value * 7 % 23) as f32 * 0.11) - 1.2).collect();
        let bias: Vec<f32> = (0..columns).map(|value| value as f32 * 0.25 - 0.5).collect();

        let cpu = crate::transformer::project_on_cpu(&left, rows, inner, columns, &right);
        let gpu = super::matmul_nt(&left, &right, None, rows, columns, inner).unwrap();
        for (cpu, gpu) in cpu.iter().zip(&gpu) { assert!((cpu - gpu).abs() < 1e-4, "CPU {cpu}, GPU {gpu}"); }

        let biased = super::matmul_nt(&left, &right, Some(&bias), rows, columns, inner).unwrap();
        for (index, value) in biased.iter().enumerate() { assert!((value - (cpu[index] + bias[index % columns])).abs() < 1e-4); }

        let out_gradient: Vec<f32> = (0..rows * columns).map(|value| ((value * 5 % 17) as f32 * 0.09) - 0.7).collect();
        let mut cpu_weight = vec![0.0; right.len()];
        let mut cpu_input = vec![0.0; left.len()];
        crate::transformer::project_backward_on_cpu(&left, &out_gradient, rows, inner, columns, &right, &mut cpu_weight, &mut cpu_input);
        let (gpu_input, gpu_weight) = super::matmul_nt_backward(&out_gradient, &left, &right, rows, columns, inner).unwrap();
        for (cpu, gpu) in cpu_input.iter().zip(&gpu_input) { assert!((cpu - gpu).abs() < 1e-4, "input CPU {cpu}, GPU {gpu}"); }
        for (cpu, gpu) in cpu_weight.iter().zip(&gpu_weight) { assert!((cpu - gpu).abs() < 1e-4, "weight CPU {cpu}, GPU {gpu}"); }

        let sums = super::column_sum(&out_gradient, rows, columns).unwrap();
        for column in 0..columns {
            let expected: f32 = (0..rows).map(|row| out_gradient[row * columns + column]).sum();
            assert!((sums[column] - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn transformer_block_gradients_match_between_cpu_and_cuda() {
        // Large enough to cross the CUDA dispatch threshold inside the block.
        let model = Model::language_model(64, 128, 4, 512, 1, 32).unwrap();
        let tokens: Vec<u32> = (0..24).map(|value| (value * 7 % 64) as u32).collect();
        let targets: Vec<u32> = (0..24).map(|value| (value * 11 % 64) as u32).collect();

        super::set_enabled(false);
        let (cpu_loss, cpu) = model.language_model_step(&tokens, &targets).unwrap();
        super::set_enabled(true);
        let (gpu_loss, gpu) = model.language_model_step(&tokens, &targets).unwrap();
        super::set_enabled(false);

        assert!((cpu_loss - gpu_loss).abs() < 1e-3, "loss CPU {cpu_loss}, GPU {gpu_loss}");
        for (index, (cpu_layer, gpu_layer)) in cpu.iter().zip(&gpu).enumerate() {
            let (Some(cpu_layer), Some(gpu_layer)) = (cpu_layer, gpu_layer) else { continue };
            for (cpu, gpu) in cpu_layer.weights.to_host().iter().zip(&gpu_layer.weights.to_host()) { assert!((cpu - gpu).abs() < 1e-4, "layer {index} weight CPU {cpu}, GPU {gpu}"); }
            for (cpu, gpu) in cpu_layer.biases.to_host().iter().zip(&gpu_layer.biases.to_host()) { assert!((cpu - gpu).abs() < 1e-4, "layer {index} bias CPU {cpu}, GPU {gpu}"); }
        }
    }

    #[test]
    fn cached_parameters_refresh_when_weights_change_in_place() {
        super::set_enabled(true);
        let (rows, columns, inner) = (4usize, 3usize, 5usize);
        // Non-zero throughout, so perturbing right[0] provably changes out[0].
        let left: Vec<f32> = (0..rows * inner).map(|value| value as f32 * 0.1 + 1.0).collect();
        let mut right: Vec<f32> = (0..columns * inner).map(|value| value as f32 * 0.2 + 0.5).collect();

        let first = super::matmul_nt(&left, &right, None, rows, columns, inner).unwrap();
        super::matmul_nt(&left, &right, None, rows, columns, inner).unwrap();
        // The epoch is global, so a parallel test may invalidate between calls; what must hold is
        // that stale entries are evicted rather than accumulating.
        assert!(super::cached_parameter_count() <= 4, "device cache grew past the live parameter set");

        // Same buffer, same length, mutated contents: a cache keyed on the address alone would
        // silently return the previous result here.
        right[0] += 5.0;
        super::invalidate_device_cache();
        let second = super::matmul_nt(&left, &right, None, rows, columns, inner).unwrap();
        let expected = crate::transformer::project_on_cpu(&left, rows, inner, columns, &right);
        for (actual, expected) in second.iter().zip(&expected) { assert!((actual - expected).abs() < 1e-4, "stale cache: {actual} vs {expected}"); }
        assert!((first[0] - second[0]).abs() > 1e-3, "mutating a weight did not change the result");
        super::set_enabled(false);
    }

    #[test]
    fn training_updates_are_visible_to_the_cached_gpu_path() {
        // A full training step mutates every weight in place; the next forward pass must see them.
        let mut model = Model::language_model(48, 128, 4, 512, 1, 16).unwrap();
        let tokens: Vec<u32> = (0..12).map(|value| (value * 5 % 48) as u32).collect();
        let targets: Vec<u32> = (0..12).map(|value| (value * 7 % 48) as u32).collect();
        let interrupted = std::sync::atomic::AtomicBool::new(false);

        super::set_enabled(true);
        let before = model.language_model_step(&tokens, &targets).unwrap().0;
        model.train_language_model(&[(tokens.clone(), targets.clone())], 12, 0.01, LearningFunction::Adam, &interrupted, |_, _| {}).unwrap();
        let gpu_after = model.language_model_step(&tokens, &targets).unwrap().0;
        super::set_enabled(false);
        let cpu_after = model.language_model_step(&tokens, &targets).unwrap().0;

        assert!(gpu_after < before, "training did not reduce loss: {before} -> {gpu_after}");
        assert!((gpu_after - cpu_after).abs() < 1e-3, "GPU saw stale weights: {gpu_after} vs CPU {cpu_after}");
    }

    #[test]
    fn device_resident_adamw_steps_match_cpu_over_a_long_run() {
        // Moments stay on the device across all twenty steps; drift here would compound silently.
        let size = 64;
        let mut cpu_parameters: Vec<f32> = (0..size).map(|value| (value as f32 * 0.037).sin()).collect();
        let mut gpu_parameters = cpu_parameters.clone();
        let (mut cpu_first, mut cpu_second) = (vec![0.0; size], vec![0.0; size]);
        let (mut gpu_first, mut gpu_second) = (vec![0.0; size], vec![0.0; size]);
        let (rate, decay) = (0.01f32, 0.01f32);

        for step in 1..=20u32 {
            let gradients: Vec<f32> = (0..size).map(|value| ((value as f32 + step as f32) * 0.11).cos() * 0.3).collect();
            for (index, parameter) in cpu_parameters.iter_mut().enumerate() {
                LearningFunction::AdamW { weight_decay: decay }.update(parameter, gradients[index], rate, &mut cpu_first[index], &mut cpu_second[index], step);
            }
            super::adamw_step(&mut gpu_parameters, &gradients, &mut gpu_first, &mut gpu_second, rate, decay, step).unwrap();
            for (cpu, gpu) in cpu_parameters.iter().zip(&gpu_parameters) {
                assert!((cpu - gpu).abs() < 1e-5, "step {step}: CPU {cpu}, GPU {gpu}");
            }
        }

        // Moments were never copied back during the loop, so the host buffers are still zero.
        assert!(gpu_first.iter().all(|value| *value == 0.0), "moments left the device early");
        super::flush_moments(&mut gpu_first, &mut gpu_second).unwrap();
        for (cpu, gpu) in cpu_first.iter().zip(&gpu_first) { assert!((cpu - gpu).abs() < 1e-6, "first moment CPU {cpu}, GPU {gpu}"); }
        for (cpu, gpu) in cpu_second.iter().zip(&gpu_second) { assert!((cpu - gpu).abs() < 1e-6, "second moment CPU {cpu}, GPU {gpu}"); }
        super::clear_moment_buffers();
    }

    #[test]
    fn device_gradient_adamw_matches_cpu_over_a_long_run() {
        let size = 64;
        let mut cpu_parameters: Vec<f32> = (0..size).map(|value| (value as f32 * 0.037).sin()).collect();
        let mut gpu_parameters = cpu_parameters.clone();
        let (mut cpu_first, mut cpu_second) = (vec![0.0; size], vec![0.0; size]);
        let (mut gpu_first, mut gpu_second) = (vec![0.0; size], vec![0.0; size]);
        let (rate, decay) = (0.01f32, 0.01f32);

        for step in 1..=20u32 {
            let gradients: Vec<f32> = (0..size).map(|value| ((value as f32 + step as f32) * 0.11).cos() * 0.3).collect();
            for (index, parameter) in cpu_parameters.iter_mut().enumerate() {
                LearningFunction::AdamW { weight_decay: decay }.update(parameter, gradients[index], rate, &mut cpu_first[index], &mut cpu_second[index], step);
            }
            let device_gradient = super::device_upload(&gradients).unwrap();
            super::adamw_step_device(&gpu_parameters, &device_gradient, &mut gpu_first, &mut gpu_second, rate, decay, step).unwrap();
            super::flush_parameters(&mut gpu_parameters).unwrap();
            for (cpu, gpu) in cpu_parameters.iter().zip(&gpu_parameters) {
                assert!((cpu - gpu).abs() < 1e-5, "step {step}: CPU {cpu}, GPU {gpu}");
            }
        }
        super::clear_moment_buffers();
    }

    #[test]
    fn dense_cuda_chain_rejects_graphs_the_kernel_cannot_handle() {
        assert!(Model::mnist().dense_cuda_chain().is_none());
        assert!(Model::mnist_memory_bank(8, 4, 2).unwrap().dense_cuda_chain().is_none());
        assert!(Model::dense(vec![4, 3, 2], 2).unwrap().dense_cuda_chain().is_none());
        assert!(Model::dense(vec![4, 3, 2], 0).unwrap().dense_cuda_chain().is_some());
    }

    #[test]
    fn convolution_and_pooling_match_cpu_reference() {
        let input = Tensor::new(1, 4, 4, (1..=16).map(|value| value as f32).collect()).unwrap();
        let weights = vec![0.25; 9]; let biases = vec![0.5];
        let cpu_convolution = cpu_conv2d(&input, 1, 1, 3, &weights, &biases).unwrap();
        let gpu_convolution = super::conv2d(&input, 1, 3, &weights, &biases).unwrap();
        assert_eq!(cpu_convolution.values, gpu_convolution.values);
        let cpu_pool = cpu_max_pool2d(&input, 2).unwrap();
        let gpu_pool = super::max_pool2d(&input, 2).unwrap();
        assert_eq!(cpu_pool.values, gpu_pool.values);
    }

    #[test]
    fn batched_convolution_and_pooling_match_cpu_reference() {
        let input = BatchTensor::new(2, 1, 3, 3, (1..=18).map(|value| value as f32).collect()).unwrap();
        let weights = vec![1.0; 4]; let biases = vec![0.0];
        assert_eq!(cpu_batch_conv2d(&input, 1, 2, &weights, &biases).unwrap(), super::batch_conv2d(&input, 1, 2, &weights, &biases).unwrap());
        assert_eq!(cpu_batch_max_pool2d(&input, 3).unwrap(), super::batch_max_pool2d(&input, 3).unwrap());
    }

    #[test]
    fn batched_relu_and_pool_backward_match_cpu_reference() {
        let input = BatchTensor::new(2, 1, 2, 2, vec![-1.0, 2.0, 3.0, 0.0, 4.0, -2.0, 1.0, 5.0]).unwrap();
        let gradient = BatchTensor::new(2, 1, 2, 2, vec![1.0; 8]).unwrap();
        assert_eq!(cpu_batch_relu_backward(&input, &gradient).unwrap(), super::batch_relu_backward(&input, &gradient).unwrap());
        let pooled_gradient = BatchTensor::new(2, 1, 1, 1, vec![2.0, 3.0]).unwrap();
        assert_eq!(cpu_batch_pool_backward(&input, 2, &pooled_gradient).unwrap(), super::batch_max_pool2d_backward(&input, 2, &pooled_gradient).unwrap());
    }

    #[test]
    fn adamw_update_matches_cpu_reference() {
        let mut cpu_parameters = vec![0.5, -0.25]; let mut gpu_parameters = cpu_parameters.clone();
        let gradients = vec![0.2, -0.4]; let mut cpu_first = vec![0.0; 2]; let mut gpu_first = cpu_first.clone(); let mut cpu_second = vec![0.0; 2]; let mut gpu_second = cpu_second.clone();
        for ((parameter, gradient), (first, second)) in cpu_parameters.iter_mut().zip(&gradients).zip(cpu_first.iter_mut().zip(cpu_second.iter_mut())) { LearningFunction::AdamW { weight_decay: 0.01 }.update(parameter, *gradient, 0.001, first, second, 1); }
        super::adamw_update(&mut gpu_parameters, &gradients, &mut gpu_first, &mut gpu_second, 0.001, 0.01, 1).unwrap();
        for (cpu, gpu) in cpu_parameters.iter().zip(&gpu_parameters).chain(cpu_first.iter().zip(&gpu_first)).chain(cpu_second.iter().zip(&gpu_second)) { assert!((cpu - gpu).abs() < 1e-6, "CPU {cpu}, GPU {gpu}"); }
    }

    #[test]
    fn cnn_backpropagation_matches_cpu_reference() {
        let cnn = Model::with_input_shape(1, 3, 3, vec![
            CnnLayer::Conv2d { in_channels: 1, out_channels: 1, kernel_size: 2, weights: vec![0.2, -0.1, 0.3, 0.4], biases: vec![0.1] },
            CnnLayer::Relu,
            CnnLayer::MaxPool2d { kernel_size: 2 },
            CnnLayer::Flatten,
            CnnLayer::Dense { inputs: 1, outputs: 2, weights: vec![0.5, -0.25], biases: vec![0.0, 0.1] },
        ]);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let target = vec![1.0, 0.0];
        let cpu = cnn.cpu_sample_gradients(&input, &target).unwrap();
        let gpu = cnn.cuda_sample_gradients(&input, &target).unwrap();
        for (cpu, gpu) in cpu.iter().zip(&gpu) {
            if let (Some(cpu), Some(gpu)) = (cpu, gpu) {
                for (cpu, gpu) in cpu.weights.to_host().into_iter().zip(gpu.weights.to_host()).chain(cpu.biases.to_host().into_iter().zip(gpu.biases.to_host())) {
                    assert!((cpu - gpu).abs() < 1e-5, "CPU {cpu}, GPU {gpu}");
                }
            }
        }
    }
}

pub fn probe() -> Result<String, String> {
    let device_count = CudaContext::device_count().map_err(|error| format!("CUDA device count: {error}"))?;
    let device = CudaContext::new(0).map_err(|error| format!("CUDA device 0: {error}"))?;
    let device_name = device.name().map_err(|error| format!("CUDA device 0 name: {error}"))?;
    Ok(format!("CUDA devices available: {device_count}; device 0 ready: {device_name}"))
}
