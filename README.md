# QuickLog

Minimal Python package for comfortable ad-hoc logging and visualization of scalar values in ML experiments.

## Installation

```bash
pip install git+https://github.com/tobiaaa/QuickLog.git
```

## Quick Start

```python
from quicklog import QuickLog

log = QuickLog()

for epoch in range(100):
    loss = train_step()
    log(loss=loss, accuracy=accuracy)

# Visualization happens automatically on exit
```

## Usage

### Logging Values

Two calling styles are supported:

```python
# Positional arguments
log("loss", 0.5)

# Keyword arguments (multiple metrics at once)
log(loss=0.5, accuracy=0.92, lr=0.001)

# Explicit step
log("loss", 0.3, step=100)
```

### Logging Images

Use `log.image()` to log matrices and images:

```python
# Log attention weights or feature maps
log.image("attention", attention_weights)

# With visualization options
log.image("features", feature_map, cmap="hot", vmin=0, vmax=1)

# Keyword arguments (multiple images at once)
log.image(attention=attn_weights, features=feature_map)

# Explicit step
log.image("activations", layer_output, step=epoch)
```

Supported formats:
- 2D arrays: Displayed as grayscale with colormap
- 3D arrays (H, W, 3) or (H, W, 4): RGB/RGBA images
- 3D arrays (3, H, W) or (4, H, W): Auto-transposed to HWC format (common for PyTorch)

### Logging Histograms

Use `log.histogram()` to accumulate values and visualize their distribution:

```python
# Log individual values during training
for batch in dataloader:
    log.histogram("grad_norm", gradient.norm())

# Or log entire arrays at once
log.histogram("weights", model.layer.weight.flatten())

# Keyword arguments (multiple histograms at once)
log.histogram(gradients=grads, activations=acts)

# Custom bin count
log.histogram("loss_dist", losses, bins=100)
```

Values accumulate across calls and are rendered as histograms at exit.

### Supported Types

**Scalars:** QuickLog automatically converts various numeric types to Python floats:

- Python `int` and `float`
- NumPy scalars and arrays (shape 1x...x1)
- PyTorch tensors, including CUDA (shape 1x...x1)

```python
log(loss=torch.tensor(0.5))                 # 0-d tensor
log(loss=torch.tensor([[[0.5]]]).cuda())    # 1x1x1 CUDA tensor
log(loss=model_output.mean())               # common pattern
```

Non-scalar values raise an error to catch bugs early.

**Images:** Both NumPy arrays and PyTorch tensors are supported:

```python
log.image("heatmap", numpy_array)           # NumPy array
log.image("features", tensor.cuda())        # CUDA tensor (auto-detached)
```

### Configuration

Configure via Python or environment variables (priority: method > env var > default):

```python
log.configure(
    output="file",       # "file" (default), "show", or "both"
    layout="subplots",   # "subplots" (default), "separate", or "overlaid"
    path="metrics.png",  # output path (default: "quicklog.png")
    dpi=150,             # image resolution (default: 150)
    smoothing=0.6,       # EMA smoothing factor, 0-1 (default: 0 = none)
    image_cmap="viridis",   # colormap for 2D arrays (default: "viridis")
    image_colorbar=True,    # show colorbar for images (default: True)
    image_history=1,        # images to keep per name (default: 1)
    histogram_bins=50,      # default bins for histograms (default: 50)
)
```

Or via environment variables:

```bash
export QUICKLOG_OUTPUT=both
export QUICKLOG_LAYOUT=overlaid
export QUICKLOG_PATH=results.pdf
export QUICKLOG_DPI=300
export QUICKLOG_SMOOTHING=0.6
export QUICKLOG_IMAGE_CMAP=hot
export QUICKLOG_IMAGE_COLORBAR=True
export QUICKLOG_IMAGE_HISTORY=5
export QUICKLOG_HISTOGRAM_BINS=50
```

### Layout Modes

- **subplots**: Grid of subplots, one per metric/image/histogram (default)
- **separate**: Individual files per metric/image/histogram (`quicklog_loss.png`, `quicklog_attention.png`, ...)
- **overlaid**: All scalars on a single plot with legend; images and histograms in separate grids (`quicklog_images.png`, `quicklog_histograms.png`)

## Design

- **Singleton**: One instance per runtime, accessible from anywhere
- **Auto-visualization**: Plots are generated automatically on exit via `atexit`
- **Zero config**: Works out of the box with sensible defaults

## License

MIT