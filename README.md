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

### Supported Types

QuickLog automatically converts various numeric types to Python floats:

- Python `int` and `float`
- NumPy scalars and arrays (shape 1x...x1)
- PyTorch tensors, including CUDA (shape 1x...x1)

```python
log(loss=torch.tensor(0.5))                 # 0-d tensor
log(loss=torch.tensor([[[0.5]]]).cuda())    # 1x1x1 CUDA tensor
log(loss=model_output.mean())               # common pattern
```

Non-scalar values raise an error to catch bugs early.

### Configuration

Configure via Python or environment variables (priority: method > env var > default):

```python
log.configure(
    output="file",       # "file" (default), "show", or "both"
    layout="subplots",   # "subplots" (default), "separate", or "overlaid"
    path="metrics.png",  # output path (default: "quicklog.png")
    dpi=150,             # image resolution (default: 150)
)
```

Or via environment variables:

```bash
export QUICKLOG_OUTPUT=both
export QUICKLOG_LAYOUT=overlaid
export QUICKLOG_PATH=results.pdf
export QUICKLOG_DPI=300
```

### Layout Modes

- **subplots**: Grid of subplots, one per metric (default)
- **separate**: Individual files per metric (`quicklog_loss.png`, `quicklog_accuracy.png`, ...)
- **overlaid**: All metrics on a single plot with legend

## Design

- **Singleton**: One instance per runtime, accessible from anywhere
- **Auto-visualization**: Plots are generated automatically on exit via `atexit`
- **Zero config**: Works out of the box with sensible defaults

## License

MIT