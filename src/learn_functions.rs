use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningFunction {
	Sgd,
	Momentum { momentum: f32 },
	Adam,
	AdamW { weight_decay: f32 },
}

impl Default for LearningFunction {
	fn default() -> Self { Self::Sgd }
}

impl LearningFunction {
	pub fn update(self, parameter: &mut f32, gradient: f32, learning_rate: f32, first_moment: &mut f32, second_moment: &mut f32, step: u32) {
		match self {
			Self::Sgd => *parameter -= learning_rate * gradient,
			Self::Momentum { momentum } => {
				*first_moment = momentum * *first_moment - learning_rate * gradient;
				*parameter += *first_moment;
			}
			Self::Adam | Self::AdamW { .. } => {
				let beta1 = 0.9;
				let beta2 = 0.999;
				*first_moment = beta1 * *first_moment + (1.0 - beta1) * gradient;
				*second_moment = beta2 * *second_moment + (1.0 - beta2) * gradient * gradient;
				let corrected_first = *first_moment / (1.0 - beta1.powi(step as i32));
				let corrected_second = *second_moment / (1.0 - beta2.powi(step as i32));
				if let Self::AdamW { weight_decay } = self { *parameter *= 1.0 - learning_rate * weight_decay; }
				*parameter -= learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8);
			}
		}
	}
}
