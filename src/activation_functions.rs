mod activation_functions;

const e: f32 = 2.718281828459045;

// Binary sigmoid activation function
pub fn binary_sigmoid(x: f32) -> f32 {
    x / (1 + e.powf(-x))
}

// ReLU (Rectified Linear Unit) activation function
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

// Randomized ReLU (RReLU) activation function
pub fn rrelu(x: f32, lower: f32, upper: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        let alpha = lower + (upper - lower) * rand::random::<f32>();
        alpha * x
    }
}

// Leaky ReLU activation function
pub fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

// Parametric ReLU (PReLU) activation function
pub fn prelu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

// Exponential Linear Unit (ELU) activation function
pub fn elu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * (e.powf(x) - 1.0)
    }
}

// Softmax activation function
pub fn softmax(x: f32) -> f32 {
    e.powf(x) / (1.0 + e.powf(x))
}

// Hyperbolic tangent (tanh) activation function
pub fn tanh(x: f32) -> f32 {
    (e.powf(x) - e.powf(-x)) / (e.powf(x) + e.powf(-x))
}

// Gaussian Error Linear Unit (GELU) activation function
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x / (2.0_f32).sqrt()).tanh())
}