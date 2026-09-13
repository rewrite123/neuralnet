//! Dynamic growth operations for transformer models.
//!
//! Every operation here is function preserving: the model's output immediately after growing is
//! identical to its output before. That matters because a growth step that perturbs the function
//! forces training to recover ground it has already covered, which would confound any comparison
//! between growth strategies.

use crate::{
    model::{Layer, Model},
    transformer::BlockShape,
};
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum GrowthStrategy {
    /// Activate Together Grow Together: chaos-driven placement, near-passthrough initialisation.
    Atgt,
    /// Artificial neurogenesis: split the highest-importance units into new ones.
    Ang,
    /// MixtureGrowth: new units are recombinations of existing parameter templates.
    Mixture,
}

/// Where a block's packed weights live, for a given feed-forward width.
struct BlockLayout {
    square: usize,
    feed: usize,
    d_model: usize,
    ff_hidden: usize,
}

impl BlockLayout {
    fn new(d_model: usize, ff_hidden: usize) -> Self {
        Self { square: d_model * d_model, feed: ff_hidden * d_model, d_model, ff_hidden }
    }
    fn w1_start(&self) -> usize { 4 * self.square }
    fn w2_start(&self) -> usize { self.w1_start() + self.feed }
    fn gamma_start(&self) -> usize { self.w2_start() + self.feed }
}

/// Widens a block's feed-forward network by `extra` hidden units.
///
/// New `w2` columns are zero, so the block's output is unchanged no matter how the new `w1` rows
/// are initialised. The strategy therefore only controls where the new units start searching from.
pub fn grow_feed_forward(model: &mut Model, block_index: usize, extra: usize, strategy: GrowthStrategy) -> Result<(), String> {
    if extra == 0 { return Ok(()); }
    let Some(Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases }) = model.layers.get_mut(block_index) else {
        return Err(format!("layer {block_index} is not a transformer block"));
    };
    let (d_model, heads) = (*d_model, *heads);
    let old = BlockLayout::new(d_model, *ff_hidden);
    let new_hidden = *ff_hidden + extra;
    let new = BlockLayout::new(d_model, new_hidden);
    let mut rng = rand::rng();

    let mut grown_weights = vec![0.0; 4 * new.square + 2 * new.feed + 2 * d_model];
    grown_weights[..old.w1_start()].copy_from_slice(&weights[..old.w1_start()]);

    // w1 is [ff_hidden][d_model]: append whole rows.
    let old_w1 = &weights[old.w1_start()..old.w2_start()];
    grown_weights[new.w1_start()..new.w1_start() + old.feed].copy_from_slice(old_w1);
    for unit in 0..extra {
        let row = new.w1_start() + (old.ff_hidden + unit) * d_model;
        let source = new_unit_source(strategy, old.ff_hidden, unit, &mut rng);
        for column in 0..d_model {
            grown_weights[row + column] = match source {
                UnitSource::Noise(scale) => rng.random_range(-scale..scale),
                UnitSource::Copy(index) => old_w1[index * d_model + column],
                UnitSource::Blend(left, right, mix) => old_w1[left * d_model + column] * mix + old_w1[right * d_model + column] * (1.0 - mix),
            };
        }
    }

    // w2 is [d_model][ff_hidden]: every row gains `extra` trailing entries, left at zero.
    let old_w2 = &weights[old.w2_start()..old.gamma_start()];
    for row in 0..d_model {
        let target = new.w2_start() + row * new_hidden;
        grown_weights[target..target + old.ff_hidden].copy_from_slice(&old_w2[row * old.ff_hidden..(row + 1) * old.ff_hidden]);
    }

    grown_weights[new.gamma_start()..].copy_from_slice(&weights[old.gamma_start()..]);

    // b1 gains one entry per new unit; b2 and both layer-norm shifts are unchanged.
    let mut grown_biases = vec![0.0; new_hidden + 3 * d_model];
    grown_biases[..old.ff_hidden].copy_from_slice(&biases[..old.ff_hidden]);
    grown_biases[new_hidden..].copy_from_slice(&biases[old.ff_hidden..]);

    *weights = grown_weights;
    *biases = grown_biases;
    if let Some(Layer::TransformerBlock { ff_hidden, .. }) = model.layers.get_mut(block_index) { *ff_hidden = new_hidden; }
    let _ = heads;
    model.invalidate_after_growth();
    Ok(())
}

enum UnitSource {
    Noise(f32),
    Copy(usize),
    Blend(usize, usize, f32),
}

fn new_unit_source(strategy: GrowthStrategy, existing: usize, unit: usize, rng: &mut rand::rngs::ThreadRng) -> UnitSource {
    match strategy {
        // Near-passthrough: small random weights that training shapes from scratch.
        GrowthStrategy::Atgt => UnitSource::Noise(0.02),
        // Neurogenesis: clone an existing unit so the new one starts in a useful region.
        GrowthStrategy::Ang => UnitSource::Copy(unit % existing.max(1)),
        // Template recombination: blend two existing units.
        GrowthStrategy::Mixture => {
            let left = rng.random_range(0..existing.max(1));
            let right = rng.random_range(0..existing.max(1));
            UnitSource::Blend(left, right, rng.random_range(0.25..0.75))
        }
    }
}

/// Removes `extra` hidden units from a transformer block while preserving its function.
///
/// The shrink succeeds only when the trailing units have zero output projection weights. Those
/// units make no contribution to the residual branch, so removing them is exactly function
/// preserving rather than an arbitrary trained-neuron deletion.
pub fn shrink_feed_forward(model: &mut Model, block_index: usize, extra: usize, strategy: GrowthStrategy) -> Result<(), String> {
    if extra == 0 { return Ok(()); }
    let Some(Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases }) = model.layers.get_mut(block_index) else {
        return Err(format!("layer {block_index} is not a transformer block"));
    };
    let (d_model, heads) = (*d_model, *heads);
    let old = BlockLayout::new(d_model, *ff_hidden);
    let new_hidden = old.ff_hidden.saturating_sub(extra).max(2);
    if new_hidden == 0 || new_hidden >= old.ff_hidden { return Ok(()); }
    let new = BlockLayout::new(d_model, new_hidden);
    let old_w1 = &weights[old.w1_start()..old.w2_start()];
    let old_w2 = &weights[old.w2_start()..old.gamma_start()];
    if old_w2.chunks_exact(old.ff_hidden).any(|row| row[new_hidden..].iter().any(|weight| *weight != 0.0)) {
        return Ok(());
    }
    let mut shrunk_weights = vec![0.0; 4 * new.square + 2 * new.feed + 2 * d_model];
    shrunk_weights[..old.w1_start()].copy_from_slice(&weights[..old.w1_start()]);
    for row in 0..new_hidden {
        let source = row * d_model;
        let target = new.w1_start() + row * d_model;
        shrunk_weights[target..target + d_model].copy_from_slice(&old_w1[source..source + d_model]);
    }
    for row in 0..d_model {
        let target = new.w2_start() + row * new_hidden;
        let source = row * old.ff_hidden;
        shrunk_weights[target..target + new_hidden].copy_from_slice(&old_w2[source..source + new_hidden]);
    }
    shrunk_weights[new.gamma_start()..].copy_from_slice(&weights[old.gamma_start()..]);
    let mut shrunk_biases = vec![0.0; new_hidden + 3 * d_model];
    shrunk_biases[..new_hidden].copy_from_slice(&biases[..new_hidden]);
    shrunk_biases[new_hidden..].copy_from_slice(&biases[old.ff_hidden..]);
    *weights = shrunk_weights;
    *biases = shrunk_biases;
    *ff_hidden = new_hidden;
    let _ = heads;
    let _ = strategy;
    model.invalidate_after_growth();
    Ok(())
}

/// Inserts a transformer block after `after_index` that is exactly the identity.
///
/// The attention output projection and the feed-forward output weights are zero, so both residual
/// branches contribute nothing until training moves them.
pub fn insert_block(model: &mut Model, after_index: usize, strategy: GrowthStrategy) -> Result<(), String> {
    let Some(Layer::TransformerBlock { d_model, heads, ff_hidden, .. }) = model.layers.get(after_index) else {
        return Err(format!("layer {after_index} is not a transformer block"));
    };
    let (d_model, heads, ff_hidden) = (*d_model, *heads, *ff_hidden);
    let shape = BlockShape { sequence: 0, d_model, heads, ff_hidden };
    let layout = BlockLayout::new(d_model, ff_hidden);
    let mut rng = rand::rng();

    let mut weights = vec![0.0; shape.weight_count()];
    // Query, key, and value projections are free to be non-zero: the zero output projection
    // discards whatever they produce until training says otherwise.
    let scale = (2.0 / d_model as f32).sqrt();
    let source: Vec<f32> = match strategy {
        GrowthStrategy::Ang => match model.layers.get(after_index) {
            Some(Layer::TransformerBlock { weights, .. }) => weights[..3 * layout.square].to_vec(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    for index in 0..3 * layout.square {
        weights[index] = if source.is_empty() { rng.random_range(-scale..scale) } else { source[index] };
    }
    for index in 0..layout.feed {
        weights[layout.w1_start() + index] = rng.random_range(-scale..scale);
    }
    // Layer-norm gains must be one for the normalisation to be a no-op scale.
    for index in layout.gamma_start()..weights.len() {
        weights[index] = 1.0;
    }

    let biases = vec![0.0; shape.bias_count()];
    model.layers.insert(after_index + 1, Layer::TransformerBlock { d_model, heads, ff_hidden, weights, biases });
    model.invalidate_after_growth();
    Ok(())
}

/// Widens a memory bank by `extra` slots, leaving the new slots' contribution at zero.
pub fn grow_memory_bank(model: &mut Model, layer_index: usize, extra: usize) -> Result<(), String> {
    if extra == 0 { return Ok(()); }
    let Some(Layer::MemoryBank { inputs, slots, weights, biases }) = model.layers.get_mut(layer_index) else {
        return Err(format!("layer {layer_index} is not a memory bank"));
    };
    let inputs = *inputs;
    weights.extend(std::iter::repeat_n(0.0, extra * inputs));
    biases.extend(std::iter::repeat_n(0.0, extra));
    *slots += extra;
    // Every downstream reader sees a wider bank, so their input widths must grow to match.
    let added = extra;
    for layer in model.layers.iter_mut().skip(layer_index + 1) {
        match layer {
            Layer::Dense { inputs, outputs, weights, biases } => {
                let mut grown = vec![0.0; (*inputs + added) * *outputs];
                for output in 0..*outputs {
                    grown[output * (*inputs + added)..output * (*inputs + added) + *inputs].copy_from_slice(&weights[output * *inputs..(output + 1) * *inputs]);
                }
                *inputs += added;
                *weights = grown;
                let _ = biases;
                break;
            }
            _ => continue,
        }
    }
    model.invalidate_after_growth();
    Ok(())
}

/// Per-layer statistics used to decide where to grow.
#[derive(Clone, Debug, Default)]
pub struct LayerSignal {
    pub index: usize,
    /// Spread of the feed-forward weights, the "chaos" the ATGT specification refers to.
    pub chaos: f32,
    /// Mean absolute magnitude of the output projection, used as an importance proxy.
    pub importance: f32,
    pub ff_hidden: usize,
}

/// Measures each transformer block's weight chaos and importance.
///
/// Chaos is the normalised Shannon entropy of the `w1` magnitude distribution, which lands in
/// [0, 1] and is scale invariant. The coefficient of variation was tried first and rejected: for
/// Gaussian weights it sits at sqrt(pi/2), about 1.25, so it saturated the growth cap on every
/// measurement and the specification's "chaos times maximum neurons" degenerated into a constant.
pub fn measure(model: &Model) -> Vec<LayerSignal> {
    const BINS: usize = 32;
    model.layers.iter().enumerate().filter_map(|(index, layer)| {
        let Layer::TransformerBlock { d_model, ff_hidden, weights, .. } = layer else { return None };
        let layout = BlockLayout::new(*d_model, *ff_hidden);
        let w1 = &weights[layout.w1_start()..layout.w2_start()];
        // Binned over signed values, not magnitudes: weights alternating between +x and -x are
        // disordered, but their magnitudes are constant and would register as perfectly calm.
        let largest = w1.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let mut histogram = [0usize; BINS];
        for value in w1 {
            let bin = if largest > 0.0 { (((value / largest) * 0.5 + 0.5) * (BINS - 1) as f32).round() as usize } else { 0 };
            histogram[bin.min(BINS - 1)] += 1;
        }
        let total = w1.len() as f32;
        let entropy: f32 = histogram.iter().filter(|count| **count > 0).map(|count| {
            let share = *count as f32 / total;
            -share * share.ln()
        }).sum();
        let w2 = &weights[layout.w2_start()..layout.gamma_start()];
        Some(LayerSignal {
            index,
            chaos: (entropy / (BINS as f32).ln()).clamp(0.0, 1.0),
            importance: w2.iter().map(|value| value.abs()).sum::<f32>() / w2.len() as f32,
            ff_hidden: *ff_hidden,
        })
    }).collect()
}

/// Decides when and where to grow, following each strategy's own trigger.
///
/// The strategies deliberately keep their own schedules rather than sharing one: a growth method
/// is a policy for when and where to add capacity as much as it is a rule for initialising it.
/// Final parameter counts are reported alongside loss so a win from simply growing more is visible.
pub struct GrowthController {
    strategy: GrowthStrategy,
    max_units: usize,
    accuracy_trigger: f32,
    patience: usize,
    shrink_trigger: f32,
    shrink_patience: usize,
    new_layer_ratio: f32,
    max_blocks: usize,
    max_ff: usize,
    consecutive: usize,
    shrink_consecutive: usize,
    previous_accuracy: Option<f32>,
    best_loss: f32,
    checks: usize,
    interval: usize,
}

impl GrowthController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(strategy: GrowthStrategy, max_units: usize, accuracy_trigger: f32, patience: usize, shrink_trigger: f32, shrink_patience: usize, new_layer_ratio: f32, max_blocks: usize, max_ff: usize, interval: usize) -> Self {
        Self { strategy, max_units, accuracy_trigger, patience, shrink_trigger, shrink_patience, new_layer_ratio, max_blocks, max_ff, consecutive: 0, shrink_consecutive: 0, previous_accuracy: None, best_loss: f32::INFINITY, checks: 0, interval }
    }

    /// Called after each validation measurement. Returns a description when the model changed.
    pub fn observe(&mut self, model: &mut Model, accuracy: f32, loss: f32) -> Result<Option<String>, String> {
        self.checks += 1;
        let (triggered, should_shrink) = match self.strategy {
            // ATGT: grow when accuracy stays high long enough and shrink after a sustained
            // validation-accuracy plateau or decline, never because accuracy is merely low.
            GrowthStrategy::Atgt => {
                if accuracy >= self.accuracy_trigger { self.consecutive += 1; } else { self.consecutive = 0; }
                let accuracy_delta = self.previous_accuracy.map(|previous| accuracy - previous);
                self.previous_accuracy = Some(accuracy);
                if accuracy_delta.is_some_and(|delta| delta <= self.shrink_trigger) {
                    self.shrink_consecutive += 1;
                } else {
                    self.shrink_consecutive = 0;
                }
                let should_shrink = self.shrink_consecutive >= self.shrink_patience;
                if should_shrink { self.shrink_consecutive = 0; }
                if self.consecutive >= self.patience {
                    self.consecutive = 0;
                    (true, should_shrink)
                } else {
                    (false, should_shrink)
                }
            }
            // Neurogenesis: add capacity when progress stalls, measured relatively so it still
            // fires late in training when absolute improvements are small.
            GrowthStrategy::Ang => {
                let improved = loss < self.best_loss * 0.99;
                if improved { self.best_loss = loss; self.consecutive = 0; } else { self.consecutive += 1; }
                if self.consecutive >= self.patience { self.consecutive = 0; (true, false) } else { (false, false) }
            }
            // MixtureGrowth: expand on a fixed cadence.
            GrowthStrategy::Mixture => (self.checks % self.interval == 0, false),
        };
        if should_shrink {
            let signals = measure(model);
            let chosen = signals.iter().max_by(|left, right| left.ff_hidden.cmp(&right.ff_hidden)).unwrap();
            if chosen.ff_hidden > 64 {
                shrink_feed_forward(model, chosen.index, 64, self.strategy)?;
                let shrunk = measure(model).into_iter().find(|signal| signal.index == chosen.index).map_or(false, |signal| signal.ff_hidden < chosen.ff_hidden);
                if shrunk {
                    return Ok(Some(format!("shrunk block at layer {} by 64 neurons after {} validation checks below {:.4} accuracy improvement (acc {:.3}, loss {:.3})", chosen.index, self.shrink_patience, self.shrink_trigger, accuracy, loss)));
                }
            }
            return Ok(None);
        }
        if !triggered { return Ok(None); }

        let signals = measure(model);
        if signals.is_empty() { return Ok(None); }
        let blocks = signals.len();

        // A large width disparity between neighbours calls for a new block between them.
        let widest = signals.iter().map(|signal| signal.ff_hidden).max().unwrap_or(1) as f32;
        let narrowest = signals.iter().map(|signal| signal.ff_hidden).min().unwrap_or(1).max(1) as f32;
        if blocks < self.max_blocks && widest / narrowest >= self.new_layer_ratio {
            let widest_index = signals.iter().max_by_key(|signal| signal.ff_hidden).unwrap().index;
            insert_block(model, widest_index, self.strategy)?;
            return Ok(Some(format!("inserted block after layer {widest_index} (width ratio {:.2})", widest / narrowest)));
        }

        let chosen = match self.strategy {
            GrowthStrategy::Atgt => signals.iter().max_by(|left, right| left.chaos.total_cmp(&right.chaos)).unwrap(),
            GrowthStrategy::Ang => signals.iter().max_by(|left, right| left.importance.total_cmp(&right.importance)).unwrap(),
            GrowthStrategy::Mixture => signals.iter().min_by_key(|signal| signal.ff_hidden).unwrap(),
        };
        // Every strategy respects the same width ceiling, so none can win by growing without bound.
        let headroom = self.max_ff.saturating_sub(chosen.ff_hidden);
        if headroom == 0 { return Ok(None); }
        // ATGT scales growth by chaos; the others add a fixed block of units.
        let units = match self.strategy {
            GrowthStrategy::Atgt => ((chosen.chaos.clamp(0.0, 1.0)) * self.max_units as f32).round() as usize,
            _ => self.max_units / 2,
        }
        .clamp(1, headroom);
        grow_feed_forward(model, chosen.index, units, self.strategy)?;
        Ok(Some(format!("widened block at layer {} by {units} units (chaos {:.3}, importance {:.4})", chosen.index, chosen.chaos, chosen.importance)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(model: &Model, tokens: &[u32]) -> Vec<f32> {
        model.last_row_logits_for_test(tokens).unwrap()
    }

    #[test]
    fn shrinking_a_feed_forward_block_preserves_the_function() {
        let tokens = vec![3u32, 1, 4, 1, 5];
        let mut model = Model::language_model(32, 16, 2, 32, 2, 8).unwrap();
        grow_feed_forward(&mut model, 2, 4, GrowthStrategy::Atgt).unwrap();
        let before = logits(&model, &tokens);
        shrink_feed_forward(&mut model, 2, 4, GrowthStrategy::Atgt).unwrap();
        let after = logits(&model, &tokens);
        assert_eq!(before.len(), after.len());
        for (before, after) in before.iter().zip(&after) {
            assert!((before - after).abs() < 1e-4, "shrink changed the function: {before} vs {after}");
        }
    }

    #[test]
    fn atgt_shrinks_after_a_sustained_accuracy_plateau() {
        let mut model = Model::language_model(32, 16, 2, 32, 1, 8).unwrap();
        grow_feed_forward(&mut model, 2, 64, GrowthStrategy::Atgt).unwrap();
        let mut controller = GrowthController::new(GrowthStrategy::Atgt, 16, 2.0, 1, 0.001, 3, 2.0, 4, 128, 3);
        assert!(controller.observe(&mut model, 0.50, 1.0).unwrap().is_none());
        assert!(controller.observe(&mut model, 0.5005, 1.0).unwrap().is_none());
        assert!(controller.observe(&mut model, 0.5010, 1.0).unwrap().is_none());
        let change = controller.observe(&mut model, 0.5015, 1.0).unwrap();
        assert!(change.is_some());
        let Layer::TransformerBlock { ff_hidden, .. } = &model.layers[2] else { panic!() };
        assert_eq!(*ff_hidden, 32);
    }

    #[test]
    fn widening_the_feed_forward_network_preserves_the_function() {
        let tokens = vec![3u32, 1, 4, 1, 5];
        for strategy in [GrowthStrategy::Atgt, GrowthStrategy::Ang, GrowthStrategy::Mixture] {
            let mut model = Model::language_model(32, 16, 2, 32, 2, 8).unwrap();
            let before = logits(&model, &tokens);
            grow_feed_forward(&mut model, 2, 24, strategy).unwrap();
            let after = logits(&model, &tokens);
            assert_eq!(before.len(), after.len());
            for (before, after) in before.iter().zip(&after) {
                assert!((before - after).abs() < 1e-4, "{strategy:?} changed the output: {before} vs {after}");
            }
            let Layer::TransformerBlock { ff_hidden, weights, biases, d_model, heads } = &model.layers[2] else { panic!() };
            assert_eq!(*ff_hidden, 56);
            let shape = BlockShape { sequence: 0, d_model: *d_model, heads: *heads, ff_hidden: *ff_hidden };
            assert_eq!(weights.len(), shape.weight_count());
            assert_eq!(biases.len(), shape.bias_count());
        }
    }

    #[test]
    fn an_inserted_block_is_the_identity() {
        let tokens = vec![2u32, 7, 1, 3];
        for strategy in [GrowthStrategy::Atgt, GrowthStrategy::Ang, GrowthStrategy::Mixture] {
            let mut model = Model::language_model(32, 16, 2, 32, 1, 8).unwrap();
            let before = logits(&model, &tokens);
            let blocks = model.layers.iter().filter(|layer| matches!(layer, Layer::TransformerBlock { .. })).count();
            insert_block(&mut model, 2, strategy).unwrap();
            assert_eq!(model.layers.iter().filter(|layer| matches!(layer, Layer::TransformerBlock { .. })).count(), blocks + 1);
            let after = logits(&model, &tokens);
            for (before, after) in before.iter().zip(&after) {
                assert!((before - after).abs() < 1e-4, "{strategy:?} inserted block was not the identity: {before} vs {after}");
            }
        }
    }

    #[test]
    fn a_grown_model_still_trains() {
        let mut model = Model::language_model(32, 16, 2, 32, 1, 8).unwrap();
        let tokens = vec![1u32, 2, 3, 4, 5];
        let targets = vec![2u32, 3, 4, 5, 6];
        let interrupted = std::sync::atomic::AtomicBool::new(false);
        model.train_language_model(&[(tokens.clone(), targets.clone())], 5, 0.01, crate::learn_functions::LearningFunction::Adam, &interrupted, |_, _| {}).unwrap();
        let before = model.language_model_step(&tokens, &targets).unwrap().0;
        grow_feed_forward(&mut model, 2, 16, GrowthStrategy::Ang).unwrap();
        insert_block(&mut model, 2, GrowthStrategy::Ang).unwrap();
        // Growth discards optimizer state, so training must restart cleanly rather than panic.
        model.train_language_model(&[(tokens.clone(), targets.clone())], 15, 0.01, crate::learn_functions::LearningFunction::Adam, &interrupted, |_, _| {}).unwrap();
        let after = model.language_model_step(&tokens, &targets).unwrap().0;
        assert!(after < before, "grown model failed to keep learning: {before} -> {after}");
    }

    #[test]
    fn chaos_ranks_a_disordered_block_above_an_ordered_one() {
        let mut model = Model::language_model(32, 16, 2, 32, 2, 8).unwrap();
        let layout = BlockLayout::new(16, 32);
        if let Layer::TransformerBlock { weights, .. } = &mut model.layers[2] {
            for index in layout.w1_start()..layout.w2_start() { weights[index] = 0.1; }
        }
        if let Layer::TransformerBlock { weights, .. } = &mut model.layers[3] {
            for (offset, index) in (layout.w1_start()..layout.w2_start()).enumerate() {
                weights[index] = if offset % 2 == 0 { 0.5 } else { -0.5 };
            }
        }
        let signals = measure(&model);
        assert_eq!(signals.len(), 2);
        assert!(signals[0].chaos < signals[1].chaos, "uniform block should be calmer: {signals:?}");
        // Chaos must stay inside [0, 1] or the specification's "chaos times maximum" saturates.
        for signal in &signals { assert!((0.0..=1.0).contains(&signal.chaos), "chaos out of range: {signal:?}"); }
        let random = Model::language_model(32, 16, 2, 32, 1, 8).unwrap();
        assert!(measure(&random)[0].chaos < 1.0, "random weights should not saturate the growth cap");
    }
}
