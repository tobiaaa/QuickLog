"""Core logging functionality for QuickLog."""
import logging

import atexit
import math
import os

_logger = logging.getLogger('QuickLog')


class QuickLog:
    """Main class for logging and visualizing scalar values in ML experiments."""

    _instance = None

    # Configuration defaults
    _defaults = {
        "output": "file",       # file, show, both
        "layout": "subplots",   # subplots, separate, overlaid
        "path": "quicklog.png",
        "dpi": 150,
        "smoothing": 0.0,       # EMA smoothing factor (0 = none, 0.9 = heavy)
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logs = {}
        self._steps = {}
        self._config = {}
        atexit.register(self._finalize)

    def configure(self, **kwargs):
        """Configure visualization options.

        Args:
            output: "file" (default), "show", or "both"
            layout: "subplots" (default), "separate", or "overlaid"
            path: Output file path (default: "quicklog.png")
            dpi: Image resolution (default: 150)
            smoothing: EMA smoothing factor, 0-1 (default: 0 = none)
        """
        valid_keys = set(self._defaults.keys())
        for key in kwargs:
            if key not in valid_keys:
                raise ValueError(f"Unknown config key: {key}")
        self._config.update(kwargs)

    def _get_config(self, key):
        """Get config value: explicit config > env var > default."""
        if key in self._config:
            return self._config[key]
        env_key = f"QUICKLOG_{key.upper()}"
        if env_key in os.environ:
            value = os.environ[env_key]
            # Coerce to same type as default
            default_type = type(self._defaults[key])
            if default_type in (int, float):
                return default_type(value)
            return value
        return self._defaults[key]

    def __call__(self, name=None, value=None, *, step=None, **kwargs):
        """Log scalar values.

        Usage:
            log("loss", 0.5)              # positional
            log(loss=0.5, accuracy=0.9)   # kwargs
            log("loss", 0.5, step=10)     # explicit step
        """
        if name is not None:
            if value is None:
                raise ValueError("Value required when using positional arguments")
            self._log(name, value, step)

        for key, val in kwargs.items():
            self._log(key, val, step)

    def _log(self, name, value, step=None):
        """Log a single metric value."""
        scalar = self._to_scalar(value)

        if name not in self.logs:
            self.logs[name] = []
            self._steps[name] = 0

        if step is None:
            step = self._steps[name]
            self._steps[name] += 1
        else:
            self._steps[name] = step + 1

        self.logs[name].append((step, scalar))

    def _to_scalar(self, value):
        """Convert various numeric types to Python float.

        Handles: int, float, numpy arrays, torch tensors (CPU and CUDA).
        Raises TypeError for non-numeric types, ValueError for non-scalar shapes.
        """
        # Python int/float: direct conversion
        if isinstance(value, (int, float)):
            return float(value)

        # Objects with .item() method (numpy/torch)
        if hasattr(value, "item"):
            # Squeeze 1x...x1 shapes to scalar
            if hasattr(value, "squeeze"):
                value = value.squeeze()

            # Check if scalar after squeeze
            if hasattr(value, "shape") and value.shape != ():
                raise ValueError(
                    f"Cannot convert to scalar: shape {value.shape} after squeeze"
                )

            return float(value.item())

        raise TypeError(
            f"Cannot convert {type(value).__name__} to scalar. "
            "Expected int, float, numpy array, or torch tensor."
        )

    def __repr__(self):
        return f"QuickLog(metrics={list(self.logs.keys())})"

    def _smooth(self, values):
        """Apply exponential moving average smoothing."""
        smoothing = self._get_config("smoothing")
        if smoothing <= 0:
            return values

        smoothed = []
        last = values[0]
        for v in values:
            last = smoothing * last + (1 - smoothing) * v
            smoothed.append(last)
        return smoothed

    def _finalize(self):
        """Called automatically at exit to visualize logged values."""
        if not self.logs:
            return
        self._visualize()

    def _visualize(self):
        """Visualize all logged metrics."""
        import matplotlib.pyplot as plt

        layout = self._get_config("layout")
        output = self._get_config("output")
        path = self._get_config("path")

        if layout == "subplots":
            self._plot_subplots(plt, path, output)
        elif layout == "separate":
            self._plot_separate(plt, path, output)
        elif layout == "overlaid":
            self._plot_overlaid(plt, path, output)

    def _plot_subplots(self, plt, path, output):
        """Plot all metrics in a grid of subplots."""
        n = len(self.logs)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

        for idx, (name, data) in enumerate(self.logs.items()):
            row, col = divmod(idx, cols)
            ax = axes[row][col]
            steps, values = zip(*data)
            ax.plot(steps, self._smooth(values))
            ax.set_title(name)
            ax.set_xlabel("step")

        # Hide unused subplots
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row][col].axis("off")

        plt.tight_layout()
        self._output(plt, fig, path, output)

    def _plot_separate(self, plt, path, output):
        """Plot each metric to a separate file."""
        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"

        for name, data in self.logs.items():
            fig, ax = plt.subplots(figsize=(6, 4))
            steps, values = zip(*data)
            ax.plot(steps, self._smooth(values))
            ax.set_title(name)
            ax.set_xlabel("step")
            plt.tight_layout()

            metric_path = f"{base}_{name}{ext}"
            self._output(plt, fig, metric_path, output)

    def _plot_overlaid(self, plt, path, output):
        """Plot all metrics on a single plot."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for name, data in self.logs.items():
            steps, values = zip(*data)
            ax.plot(steps, self._smooth(values), label=name)

        ax.set_xlabel("step")
        ax.legend()
        plt.tight_layout()
        self._output(plt, fig, path, output)

    def _output(self, plt, fig, path, output):
        """Handle output based on configuration."""
        if output in ("file", "both"):
            dpi = self._get_config("dpi")
            fig.savefig(path, dpi=dpi)
            _logger.info(f"QuickLog: saved {path}")
        if output in ("show", "both"):
            plt.show()
        plt.close(fig)
