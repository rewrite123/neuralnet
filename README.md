# neuralnet

`neuralnet` is a command-line tool for creating, training, and inspecting small neural networks. A single layer-graph engine runs every architecture — fully connected, convolutional, and memory bank — and all models are stored as GGUF v3 files with F32 tensors.

## Build

```bash
cargo build --release
```

Run the built executable as `target/release/neuralnet`, or use `cargo run --release -- <command>` while developing.

### Deploy to deeplearn

For the configured `deeplearn` SSH host, push the current committed `grownn` branch, update the remote checkout, and build the one-job CUDA release binary:

```bash
bash scripts/deploy_deeplearn.sh
```

Set `DEEPLEARN_HOST` or `DEEPLEARN_ROOT` to override the remote host or checkout path, or pass a branch name as the sole argument. The script does not stop or restart an active training process.

## Commands

### Create a model

Create a new randomly initialized dense network. Layer sizes are comma-separated, including the input and output layers.

```bash
target/release/neuralnet new --layers 784,128,10 --output models/mnist.gguf
```

For example, `--layers 4,8,3` creates a 4-input, 8-hidden-unit, 3-output classifier.

### Train a model

Train a new model using a JSON training configuration:

```bash
target/release/neuralnet train \
  --config training/mnist.json \
  --output models/mnist.gguf \
  --layers 784,128,10 \
  --learning-rate 0.01 \
  --optimizer momentum \
  --momentum 0.9
```

Defaults make the MNIST command short:

```bash
target/release/neuralnet train
```

`--train` is also accepted as a compatibility alias for `train`:

```bash
target/release/neuralnet --train --optimizer adamw --learning-rate 0.001
```

When using `cargo run`, Cargo needs its own argument separator before application flags:

```bash
cargo run --release -- --train --optimizer adamw --learning-rate 0.001
```

#### Train flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `-c, --config <PATH>` | `training/mnist.json` | Training JSON configuration to load. |
| `-o, --output <PATH>` | `models/mnist.gguf` | GGUF path for the best trained checkpoint. Parent directories are created automatically. |
| `--model <PATH>` | None | Existing GGUF model to continue training. When supplied, its architecture replaces `--layers`. |
| `--layers <SIZES>` | `784,128,10` | Comma-separated input, hidden, and output layer sizes used for a new model. |
| `--learning-rate <FLOAT>` | `0.001` | Optimizer step size. `0.001` is a suitable Adam or AdamW starting point. |
| `--optimizer <sgd\|momentum\|adam\|adamw>` | `momentum` | Parameter-update rule. |
| `--momentum <FLOAT>` | `0.9` | Momentum coefficient; used only with `--optimizer momentum`. |
| `--cuda` | Off | Run mini-batch gradients on CUDA device 0. Requires `--features gpu` and `libnvrtc.so`. Plain fully connected stacks use a batched dense kernel; other graphs use per-sample CUDA backpropagation. |
| `--epochs <N>` | config `max_epochs` | Overrides the epoch limit from the training JSON. |
| `--batch-size <N>` | config `batch_size` | Overrides the batch size from the training JSON. |
| `--max-samples <N>` | config `max_samples` | Truncates the training set, for quick smoke runs. |
| `--slots <N>` | `0` | Memory bank registers per block when creating a new model. |

The training JSON can also set `"use_cuda": true`; either that field or the `--cuda` flag enables CUDA training. The command-line flag is useful for one-off runs without editing the configuration. Training examples are shuffled before every epoch. `adam` uses first- and second-moment gradient estimates; `adamw` additionally applies decoupled weight decay of `0.01`.

Continue training an existing GGUF model with `--model`. When this option is present, `--layers` is ignored because the architecture comes from the loaded model.

```bash
target/release/neuralnet train \
  --model models/mnist.gguf \
  --config training/mnist.json \
  --output models/mnist-finetuned.gguf \
  --optimizer sgd \
  --learning-rate 0.001
```

Supported optimizers are `sgd`, `momentum`, `adam`, and `adamw`. Momentum is the default and uses `--momentum 0.9` unless changed. For MNIST, start with `--optimizer adam --learning-rate 0.001`.

When training with `momentum`, `adam`, or `adamw`, the CLI also writes `<model>.optimizer.json` next to the GGUF. It contains the optimizer moments and update count, and is loaded automatically when that GGUF is passed to `--model`. Keep the GGUF and sidecar together when resuming. Model topology edits intentionally discard the sidecar state because its tensor shapes no longer match.

### Insert a layer

`add-layer` inserts a hidden layer after an existing zero-based layer index. It cannot insert before the input layer or after the output layer. For a `784,128,10` model, `--insert-at 1` produces `784,128,<new size>,10`.

```bash
cargo run --release -- add-layer \
  --model models/mnist97.gguf \
  --output models/mnist97deep.gguf \
  --layer-size 128 \
  --insert-at 1 \
  --method passthrough
```

Methods:

| Method | Required options | Behavior |
| --- | --- | --- |
| `passthrough` | None | Preserves predictions exactly when the new size equals the preceding hidden layer. Wider layers replicate activations and split outgoing weights; narrower layers average activation groups and combine their outgoing weights. |
| `gaussian` | None | Initializes both new connections with He-scaled Gaussian weights. This changes predictions immediately and needs training. |
| `copy` | `--copy-from <CONNECTION_INDEX>` | Copies an existing connection only when its input width and requested output size match the insertion point. |
| `values` | `--values-file <PATH>` | Initializes exact input and output connections from a JSON file. |

The `values` JSON format is:

```json
{
  "input_weights": [0.0],
  "biases": [0.0],
  "output_weights": [0.0],
  "output_biases": [0.0]
}
```

`input_weights` has `new_layer_size * preceding_layer_size` values and `output_weights` has `next_layer_size * new_layer_size` values; matrices are row-major. `biases` must have `new_layer_size` values. `output_biases` is optional; omitting it preserves the existing following-layer biases.

### Edit neurons and layers

Neurons and layers can only be changed in hidden layers. Layer and neuron indexes are zero-based, with the input layer at `0` and the output layer at the final index.

Add hidden neurons at a specific index. `passthrough` inserts disconnected zero-weight neurons, preserving predictions until training changes them. `gaussian` gives new neurons He-scaled random weights and changes predictions immediately.

```bash
target/release/neuralnet add-neurons \
  --model models/mnist97deep.gguf \
  --output models/mnist97wide.gguf \
  --layer 1 \
  --insert-at 64 \
  --count 32 \
  --method passthrough
```

Remove individual hidden neurons or inclusive `[start,end]` ranges. Quote values containing brackets so the shell passes them literally:

```bash
target/release/neuralnet remove-neurons \
  --model models/mnist97wide.gguf \
  --output models/mnist97trimmed.gguf \
  --layer 1 \
  --indexes "[0,63],71"
```

The example removes neurons `0` through `63` and neuron `71`.

Remove a hidden layer with affine connection composition:

```bash
target/release/neuralnet remove-layer \
  --model models/mnist97deep.gguf \
  --output models/mnist97shallow.gguf \
  --layer 2
```

Removing a layer changes behavior in general because hidden layers use ReLU activations. The command composes the surrounding weights and biases, which is exact only where the removed ReLU behaves linearly.

For a plain fully connected stack, build with the `gpu` feature and add `--cuda` to execute mini-batch forward propagation, backpropagation, and gradient reduction on CUDA device 0 with a batched dense kernel. Any other graph shape (convolution, pooling, or memory bank layers) falls back to per-sample CUDA backpropagation, which is correct but slower. The optimizer update remains on the host. CUDA training compiles kernels at runtime, so it requires the CUDA Toolkit's `libnvrtc.so` in addition to an NVIDIA driver.

```bash
cargo run --release --features gpu -- train --cuda --config training/mnist.json
```

CNN models also support CUDA backpropagation through convolution, ReLU, max-pooling, flatten, and dense layers:

```bash
cargo run --release --features gpu -- train \
  --architecture cnn \
  --cuda \
  --model models/mnist-cnn.json \
  --output models/mnist-cnn.json \
  --optimizer adamw \
  --learning-rate 0.001
```

CNN training accepts `--epochs`, `--batch-size`, and `--max-samples`. Use a bounded smoke run before a long training job:

```bash
cargo run --release -- train \
  --architecture cnn \
  --output models/mnist-cnn.json \
  --optimizer adamw \
  --learning-rate 0.001 \
  --epochs 1 \
  --batch-size 32 \
  --max-samples 256
```

CNN training prints training accuracy after every completed epoch. Press `Ctrl+C` to stop at the next batch boundary; the CLI then asks whether to save the CNN checkpoint at the requested `--output` path or discard the interrupted changes. Accuracy is reported per epoch rather than per batch because each accuracy measurement evaluates the complete selected training set.

The CNN CUDA implementation is parity-tested against CPU gradients. It reuses one CUDA context and one compiled kernel module for the full training process. Its current limitation is per-sample device-buffer allocation and host/device transfers inside the backward pass; a future batched device-resident workspace is needed for high-throughput full-MNIST training.

### Memory bank models

`MemoryBank` is a layer type in the layer-graph engine, so it is available to any model that engine builds — convolutional or fully connected. Each `MemoryBank` layer projects its input into a fixed number of slots, appends those slots to a bank that persists for the rest of the forward pass, and emits its own activation concatenated with the whole bank. Every later block therefore reads the previous block's output *and* every register written before it, so early-layer information is preserved verbatim instead of being squeezed through each subsequent transform.

Bank geometry is set at creation time with `--hidden`, `--slots`, and `--blocks`. `--slots 0` disables banks.

| Architecture | Layers | Banks | Default geometry |
|---|---|---|---|
| `dense` | Flatten, Dense, ReLU | Optional, via `--slots` | shaped by `--layers` |
| `cnn` | Conv2d, MaxPool2d, Flatten, Dense, ReLU | Optional, via `--slots` | `--hidden 64 --slots 0 --blocks 1` |
| `memory-bank` | Flatten, Dense, ReLU, MemoryBank | Always | `--hidden 128 --slots 32 --blocks 3` |
| `transformer` | Embedding, PositionalEmbedding, TransformerBlock, LayerNorm, TimeDistributedDense | No | `--vocab 256 --hidden 128 --heads 4 --blocks 4 --max-sequence 128` |

All four share one engine, one GGUF format, and one training path, so `describe`, `evaluate`, `inspect`, checkpoint resume, and `Ctrl+C` save/discard behave identically for each.

## Transformers

`--architecture transformer` builds a decoder-only language model. Each `TransformerBlock` is pre-norm: layer normalisation, causal multi-head self-attention, a residual add, another layer normalisation, a GELU feed-forward network, and a second residual add. All of a block's matrices are packed into one weight vector (`wq`, `wk`, `wv`, `wo`, `w1`, `w2`, and the two layer-norm gains) so the existing optimizer and GGUF plumbing work unchanged.

```bash
cargo run --release -- new --architecture transformer \
  --vocab 64 --hidden 64 --heads 4 --blocks 2 --max-sequence 32 \
  --output models/tiny-lm.gguf
```

```text
0: Embedding 64 tokens x 64
1: PositionalEmbedding 32 x 64
2: TransformerBlock d_model 64, heads 4, ff 256
3: TransformerBlock d_model 64, heads 4, ff 256
4: LayerNorm 64
5: TimeDistributedDense 64->64
```

Sequences flow as `1 x sequence x d_model` tensors, and the loss is next-token cross-entropy averaged over positions. Attention is causal: position `t` may only attend to positions `0..=t`. Every parameter group is checked against finite differences in `model::tests::transformer_gradients_match_finite_differences`, and the masking is checked in `model::tests::attention_is_causal`.

Attention and the feed-forward network run on the CPU; there are no CUDA kernels for them yet.

### Importing pretrained embeddings

`import-embeddings` copies a two-dimensional matrix out of a safetensors file into a model's token embedding (`--target token`), positional embedding (`--target position`), or output head (`--target output-head`). `F32`, `F16`, and `BF16` tensors are supported. The widths must match; a larger source vocabulary is truncated.

Note that `google/embeddinggemma-300m` and `google/gemma-3-*` are **gated** on Hugging Face: fetching their weights returns `401 GatedRepo` unless you accept the licence and supply a token. `openai-community/gpt2` is ungated.

### Bootstrapping from GPT-2

GPT-2's safetensors header lists byte offsets for every tensor, so only the embedding matrices need downloading rather than the full 548 MB checkpoint:

```bash
mkdir -p data/gpt2
URL=https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
# Header length, then the header itself, gives the offsets of wte.weight and wpe.weight.
curl -sL -r 0-7 "$URL" | xxd            # first 8 bytes: little-endian header length
curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json" -o data/gpt2/vocab.json
curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt" -o data/gpt2/merges.txt
```

Pack the two tensors into `data/gpt2/embeddings.safetensors`, then build a model around them. GPT-2's embedding width fixes `--hidden 768`:

```bash
cargo run --release -- new --architecture transformer \
  --vocab 50257 --hidden 768 --heads 12 --blocks 4 --max-sequence 128 \
  --output models/gpt2-init.gguf

cargo run --release -- import-embeddings --model models/gpt2-init.gguf \
  --safetensors data/gpt2/embeddings.safetensors --tensor wte.weight --target token
cargo run --release -- import-embeddings --model models/gpt2-init.gguf \
  --safetensors data/gpt2/embeddings.safetensors --tensor wpe.weight --target position
cargo run --release -- import-embeddings --model models/gpt2-init.gguf \
  --safetensors data/gpt2/embeddings.safetensors --tensor wte.weight --target output-head
```

That is 105,684,049 parameters in a 423 MB GGUF. The embedding rows are semantically meaningful on arrival: the nearest neighbours of ` king` are ` kings`, ` King`, ` queen`, and ` prince`.

### Tokenizing, training, and generating text

The GPT-2 byte-level BPE tokenizer is implemented against the published `vocab.json` and `merges.txt`, including the byte-to-code-point table and the pre-tokenizer's contraction, letter, digit, symbol, and whitespace rules.

```bash
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/text/shakespeare.txt

cargo run --release --features gpu -- train-text --model models/gpt2-init.gguf \
  --output models/shakespeare.gguf --text data/text/shakespeare.txt \
  --sequence 128 --epochs 2 --learning-rate 0.0003 --log-every 128 --cuda

cargo run --release --features gpu -- generate --model models/shakespeare.gguf \
  --prompt "KING RICHARD III:" --tokens 50 --temperature 0.8 --top-k 40 --cuda
```

Sequences are shuffled, and progress is reported every `--log-every` sequences so a long epoch is not silent. `--temperature 0` (the default) is greedy decoding; a positive temperature samples from the top `--top-k` logits, which matters because greedy decoding on an undertrained model collapses into repetition.

### A worked example: TinyShakespeare

Starting from GPT-2 embeddings with randomly initialised transformer blocks, two epochs over the 1.1 MB TinyShakespeare corpus (338,025 tokens, 2,640 sequences of 128) took **58.6 minutes** at a sustained 192 tokens/second:

| | Loss | Perplexity |
|---|---|---|
| Start | 9.97 | 21,439 |
| End of epoch 1 | 4.45 | 86.0 |
| End of epoch 2 | 3.98 | 53.3 |

```text
KING RICHARD III:
Who is the duke?

QUEEN ELIZABETH:
Why!

KING RICHARD III:
What wits he stands by the Duke of York of York?

QUEEN MARGAR
```

The model has learned the script format, archaic vocabulary, and locally grammatical English. More tellingly, the character associations are correct per play: `KING RICHARD III` draws `QUEEN ELIZABETH`, `QUEEN MARGARET`, and "Duke of York"; `First Citizen` draws `CORIOLANUS`. It has not learned to say anything coherent, which is expected. GPT-2 saw roughly 10 billion tokens; two epochs here is 676,000, about four orders of magnitude less.

Only the embeddings carry pretrained knowledge. The blocks start random, so this trains a language model's reasoning layers from scratch on a corpus that fits in a browser tab.

```bash
cargo run --release --features gpu -- train-text --model models/gpt2-init.gguf \
  --output models/gpt2-tiny.gguf --text data/text/tiny.txt \
  --sequence 32 --epochs 25 --learning-rate 0.0003 --cuda

cargo run --release --features gpu -- generate --model models/gpt2-tiny.gguf \
  --prompt "The capital of France is" --tokens 10 --cuda
```

Only the embeddings come from GPT-2; the transformer blocks start random, so an untrained model emits a single repeated token. After training on a small corpus it reproduces it:

```text
The capital of France is Paris. The capital of Italy is Rome. The
```

**Performance.** With `--cuda`, a transformer block runs entirely on the device: layer normalisation, the query/key/value projections, causal attention with its softmax, the output projection, both residual adds, the GELU feed-forward network, and their gradients. Only the block's input crosses the bus on the way in and its output on the way out, so activations stay resident between layers instead of round-tripping per operation. Parameters are cached in device memory rather than re-uploaded per call, and greedy decoding applies the output head to the final position only. On a 105M-parameter model generating 30 tokens:

| | CPU | CUDA | Speedup |
|---|---|---|---|
| Generation | 2.54 tokens/s | 22.6 tokens/s | 8.9x |
| Training (6 steps of 32 tokens) | 22.0 s | 2.1 s | 10.5x |

Generated text is identical in both modes, and training losses match exactly (0.5595 after six steps either way).

Weight gradients stay in device memory from the backward pass through the optimizer step. `GradientData` is either a host `Vec<f32>` or a device buffer, and accumulation, scaling, and the Adam/AdamW update all work on both. The embedding table gradient is scattered on the device rather than materialised as a mostly-zero host array, and the output head's gradient never leaves the device at all.

The device cache is keyed by host address and length, and validated against a global epoch counter. Every parameter write bumps that epoch, as does every model construction and drop, so a recycled heap address can never be served a buffer belonging to freed data. Activations are deliberately never cached, because they change without bumping the epoch.

Two subtleties in that cache are worth knowing about, because both caused silently wrong training before they were fixed:

- **Aliasing.** A transformer block's weights are cached whole by the optimizer and as twelve sub-slices by the layer that consumes them. Updating the whole buffer in place left the sub-slices stale, so the forward pass kept reading pre-update weights while the loss still fell, just more slowly. Updating a parameter vector now evicts any cached buffer that aliases a sub-range of it, and blocks are flushed to the host each step so those sub-slices re-upload correctly.
- **Authority.** A buffer the optimizer has updated in place is newer than its host copy, so an epoch bump from an unrelated model must not cause it to be overwritten from the host. Such entries are re-stamped with the current epoch instead.

Parameters whose layer runs entirely on the device (the embedding table and the output head) stay resident between steps; the rest are copied back each step because their forward pass reads host memory. Optimizer moments stay resident across steps and return only when the optimizer state is stored or saved.

The CUDA enable flag is thread-local, matching the per-thread CUDA runtime and parameter cache. A process-global flag let one test thread divert another thread's arithmetic onto the GPU, which made exact-equality assertions fail intermittently.

Still on the CPU: the positional embedding lookup and the final layer normalisation, both small enough that their host round trip does not show up in the timings.

Training stores AdamW moments beside the model as `<model>.optimizer.bin` in a flat binary layout. For a 105M-parameter model that file is 845 MB; the earlier JSON encoding was 2.18 GB.

A fully connected bank model:

```bash
cargo run --release -- new --architecture memory-bank --output models/mnist-bank.json
```

```text
 0: Flatten
 1: Dense 784->128
 2: ReLU
 3: MemoryBank 128->32 slots
 4: Dense 160->128        # 128 activation + 32 bank
 5: ReLU
 6: MemoryBank 128->32 slots
 7: Dense 192->128        # 128 activation + 64 bank
 8: ReLU
 9: MemoryBank 128->32 slots
10: Dense 224->10         # 128 activation + 96 bank
```

A convolutional model with a bank head:

```bash
cargo run --release -- new --architecture cnn --slots 16 --blocks 2 --output models/mnist-convbank.json
```

This keeps the usual conv stem and replaces the dense head with bank blocks, giving `Dense 400->64`, `MemoryBank 64->16`, `Dense 80->64`, `MemoryBank 64->16`, `Dense 96->10`.

Train either one with the same flags:

```bash
cargo run --release -- train \
  --architecture memory-bank \
  --config training/mnist.json \
  --model models/mnist-bank.json \
  --output models/mnist-bank.json \
  --optimizer adamw \
  --learning-rate 0.001 \
  --epochs 20 \
  --batch-size 64
```

Bank models share the layer-graph checkpoint format with CNNs, so they work with `describe`, `evaluate`, per-epoch accuracy reporting, and `Ctrl+C` save/discard.

Because a bank slot is read by every downstream layer, backpropagation accumulates gradients for the whole bank during the reverse traversal and only finalizes a layer's slots once every later reader has contributed. This routing is verified against finite differences in `cnn::tests::memory_bank_gradients_match_finite_differences`.

Two limitations: a `MemoryBank` flattens its output, so no convolution or pooling can follow one — banks belong in the dense head. And `MemoryBank` backpropagation runs on the CPU even when `--cuda` is set; the surrounding dense and convolution layers still use CUDA.

Banks are available to the `dense` architecture too, since every architecture shares the same engine:

```bash
cargo run --release -- new --architecture dense --layers 784,128,64,10 --slots 16 --output models/mnist-bankdense.gguf
```

The model-surgery commands (`add-layer`, `add-neurons`, `remove-neurons`, `remove-layer`) reshape plain fully connected stacks only. They refuse models containing convolution, pooling, or memory bank layers rather than corrupting them, because changing a bank's width shifts the input width of every downstream layer that reads it.

During `train`, press `Ctrl+C` to stop after the current epoch. The CLI then asks whether to save the best validation checkpoint to the requested `--output` path or discard the interrupted training changes.

### Inspect a model

Validate that a GGUF file can be loaded by this tool:

```bash
target/release/neuralnet inspect models/mnist.gguf
```

### Describe a model

Print a model's layer layout, neuron count per layer, connection matrix shapes, and parameter totals:

```bash
target/release/neuralnet describe models/mnist97deep.gguf
```

### Evaluate a model

Evaluate a model once against the held-out `test` source from a training configuration:

```bash
target/release/neuralnet evaluate \
  --model models/mnist97deeptrim.gguf \
  --config training/mnist.json
```

### GPU information

The current trainer runs on CPU. CUDA and GLSL capability probing is available when built with the optional `gpu` feature:

```bash
cargo run --release --features gpu -- gpu-info
```

## MNIST Training

The checked-in [training configuration](training/mnist.json) consumes the MNIST IDX gzip archives from `data/mnist/`. It reserves 5,000 images from the 60,000-image training split for per-epoch validation. The separate `t10k` split is used only by `evaluate` for final test accuracy.

```json
{
  "max_epochs": 2000,
  "max_samples": 1000,
  "batch_size": 64,
  "validation_samples": 5000,
  "target_accuracy": 0.95
}
```

`batch_size` controls how many samples form one synchronous update. Gradients for samples in a batch are calculated in parallel across CPU worker threads, averaged, and then applied together. It defaults to `32`; use `64` or `128` for MNIST when memory permits. `target_accuracy` is a fraction from `0.0` to `1.0`. Training stops early when it is achieved. `max_epochs` is a safety cap. If the cap is reached first, the tool saves the checkpoint with the best validation accuracy instead of failing.

Set `max_samples` to `60000` to train on all MNIST training images. Training with only 1,000 samples is useful for a quick experiment, but it is unlikely to achieve 95% held-out accuracy.

## Training Configuration

A configuration has this shape:

```json
{
  "name": "example classifier",
  "max_epochs": 20,
  "max_samples": 1000,
  "batch_size": 64,
  "target_accuracy": 0.75,
  "use_cuda": false,
  "loading_functions": [],
  "training": {
    "kind": "mnist",
    "images_path": "data/mnist/train-images-idx3-ubyte.gz",
    "labels_path": "data/mnist/train-labels-idx1-ubyte.gz"
  },
  "validation": {
    "kind": "mnist",
    "images_path": "data/mnist/t10k-images-idx3-ubyte.gz",
    "labels_path": "data/mnist/t10k-labels-idx1-ubyte.gz"
  }
}
```

`training` and optional `validation` support three source kinds:

- `mnist`: Reads MNIST IDX image and label archives. Images are normalized to $[0, 1]$ and labels become 10-element one-hot vectors.
- `inline`: Uses `inputs` and `outputs` arrays directly in JSON.
- `files`: Reads two arbitrary files using named WASM loading functions. It requires `inputs`, `outputs`, `input_width`, and `output_width`.

For `files`, define each loader in `loading_functions` with a `name` and hexadecimal `wasm_hex`. A loader must export:

- `memory`
- `alloc(i32) -> i32`
- `load(i32, i32) -> i64`

The CLI copies source bytes into WASM memory at the pointer returned by `alloc`. `load(pointer, byte_length)` must return `(f32_count << 32) | output_pointer`; output is read from WASM memory as little-endian `f32` values. The declared input/output widths split those values into individual training samples.

Use the generated help for all flags:

```bash
target/release/neuralnet --help
target/release/neuralnet train --help
```
