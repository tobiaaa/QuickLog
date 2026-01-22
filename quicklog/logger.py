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
        "image_cmap": "viridis",
        "image_colorbar": True,
        "image_history": 1,     # Number of images to keep per name
        "histogram_bins": 50,
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
        self.images = {}
        self.histograms = {}
        self._steps = {}
        self._config = {}
        self._check_env_vars()
        atexit.register(self._finalize)

    def _check_env_vars(self):
        """Warn about unknown QUICKLOG_* environment variables."""
        valid_keys = {f"QUICKLOG_{k.upper()}" for k in self._defaults}

        for key in os.environ:
            if key.startswith("QUICKLOG_") and key not in valid_keys:
                valid_list = ", ".join(sorted(valid_keys))
                _logger.warning(
                    f"Unknown environment variable '{key}'. "
                    f"Valid options: {valid_list}"
                )

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

    def _to_array(self, data):
        """Convert various array types to numpy array suitable for imshow.

        Handles: numpy arrays, torch tensors (CPU and CUDA).
        Validates shape is 2D (grayscale) or 3D with 3/4 channels (RGB/RGBA).
        """
        import numpy as np

        # Handle torch tensors
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()

        # Convert to numpy if needed
        arr = np.asarray(data)

        # Squeeze singleton dimensions
        arr = np.squeeze(arr)

        # Validate shape
        if arr.ndim == 2:
            return arr  # Grayscale
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                return arr  # RGB or RGBA (H, W, C)
            elif arr.shape[0] in (3, 4):
                return np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            else:
                raise ValueError(
                    f"Invalid 3D array shape {arr.shape}. "
                    "Expected (H, W, 3), (H, W, 4), (3, H, W), or (4, H, W)."
                )
        else:
            raise ValueError(
                f"Cannot display array with {arr.ndim} dimensions. "
                "Expected 2D (grayscale) or 3D (RGB/RGBA)."
            )

    def image(self, name=None, data=None, *, step=None, cmap=None, vmin=None, vmax=None, colorbar=None, **kwargs):
        """Log image/matrix data for visualization.

        Args:
            name: Identifier for this image stream
            data: 2D array (grayscale) or 3D array (RGB/RGBA)
            step: Optional explicit step number
            cmap: Colormap (default: from config, ignored for RGB)
            vmin/vmax: Value range for colormap normalization
            colorbar: Whether to show colorbar (default: from config)

        Usage:
            log.image("attention", attention_weights)
            log.image("feature", feature_map, cmap="hot", vmin=0, vmax=1)
            log.image(attention=weights, feature=features)  # kwargs
        """
        opts = {
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "colorbar": colorbar,
        }

        if name is not None:
            if data is None:
                raise ValueError("Data required when using positional arguments")
            self._log_image(name, data, step, opts)

        for key, val in kwargs.items():
            self._log_image(key, val, step, opts)

    def _log_image(self, name, data, step, opts):
        """Log a single image."""
        array = self._to_array(data)

        if name not in self.images:
            self.images[name] = []
            if name not in self._steps:
                self._steps[name] = 0

        if step is None:
            step = self._steps[name]
            self._steps[name] += 1
        else:
            self._steps[name] = step + 1

        self.images[name].append((step, array, opts))

        # Limit history to prevent memory explosion
        max_history = self._get_config("image_history")
        if len(self.images[name]) > max_history:
            self.images[name] = self.images[name][-max_history:]

    def histogram(self, name=None, value=None, *, bins=None, **kwargs):
        """Log values for histogram visualization.

        Args:
            name: Identifier for this histogram
            value: Scalar or array of values to add to the histogram
            bins: Number of bins (default: from config)

        Usage:
            log.histogram("grad_norm", gradient.norm())       # scalar
            log.histogram("weights", model.weights.flatten()) # array
            log.histogram(activations=act, weights=w)         # kwargs
        """
        opts = {"bins": bins}

        if name is not None:
            if value is None:
                raise ValueError("Value required when using positional arguments")
            self._log_histogram(name, value, opts)

        for key, val in kwargs.items():
            self._log_histogram(key, val, opts)

    def _log_histogram(self, name, value, opts):
        """Log value(s) to a histogram. Accepts scalars or arrays."""
        if name not in self.histograms:
            self.histograms[name] = {"values": [], "opts": opts}

        # Check if it's an array-like (has shape or is a list)
        if hasattr(value, "shape") or isinstance(value, (list, tuple)):
            import numpy as np
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arr = np.asarray(value).flatten()
            self.histograms[name]["values"].extend(arr.tolist())
        else:
            scalar = self._to_scalar(value)
            self.histograms[name]["values"].append(scalar)

    def __repr__(self):
        return f"QuickLog(metrics={list(self.logs.keys())}, images={list(self.images.keys())}, histograms={list(self.histograms.keys())})"

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
        if not self.logs and not self.images and not self.histograms:
            return
        self._visualize()

    def _visualize(self):
        """Visualize all logged metrics and images."""
        import matplotlib.pyplot as plt

        layout = self._get_config("layout")
        output = self._get_config("output")
        path = self._get_config("path")

        # Collect all items to plot
        items = []
        for name, data in self.logs.items():
            items.append(("scalar", name, data))
        for name, data in self.images.items():
            items.append(("image", name, data))
        for name, data in self.histograms.items():
            items.append(("histogram", name, data))

        if not items:
            return

        if layout == "subplots":
            self._plot_subplots(plt, items, path, output)
        elif layout == "separate":
            self._plot_separate(plt, items, path, output)
        elif layout == "overlaid":
            self._plot_overlaid(plt, items, path, output)

    def _render_scalar(self, ax, name, data):
        """Render a timeseries plot on the given axes."""
        smoothing = self._get_config("smoothing")
        steps, values = zip(*data)

        if smoothing > 0:
            line, = ax.plot(steps, self._smooth(values))
            ax.plot(steps, values, alpha=0.3, color=line.get_color())
        else:
            ax.plot(steps, values)

        ax.set_title(name)
        ax.set_xlabel("step")

    def _render_image(self, ax, fig, name, data):
        """Render an image on the given axes."""
        # data = [(step, array, opts), ...]
        # Show most recent image
        step, array, opts = data[-1]

        cmap = opts.get("cmap") or self._get_config("image_cmap")
        vmin = opts.get("vmin")
        vmax = opts.get("vmax")
        colorbar = opts.get("colorbar")
        if colorbar is None:
            colorbar = self._get_config("image_colorbar")

        # Don't use colormap for RGB/RGBA images
        if array.ndim == 3:
            im = ax.imshow(array)
        else:
            im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)

        if colorbar and array.ndim == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = f"{name}" if len(data) == 1 else f"{name} (step {step})"
        ax.set_title(title)
        ax.axis("off")

    def _render_histogram(self, ax, name, data):
        """Render a histogram on the given axes."""
        values = data["values"]
        bins = data["opts"].get("bins") or self._get_config("histogram_bins")

        ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(name)
        ax.set_ylabel("count")

    def _plot_subplots(self, plt, items, path, output):
        """Plot all metrics in a grid of subplots."""
        n = len(items)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

        for idx, (item_type, name, data) in enumerate(items):
            row, col = divmod(idx, cols)
            ax = axes[row][col]

            if item_type == "scalar":
                self._render_scalar(ax, name, data)
            elif item_type == "image":
                self._render_image(ax, fig, name, data)
            elif item_type == "histogram":
                self._render_histogram(ax, name, data)

        # Hide unused subplots
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row][col].axis("off")

        plt.tight_layout()
        self._output(plt, fig, path, output)

    def _plot_separate(self, plt, items, path, output):
        """Plot each metric/image to a separate file."""
        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".png"

        for item_type, name, data in items:
            fig, ax = plt.subplots(figsize=(6, 4))

            if item_type == "scalar":
                self._render_scalar(ax, name, data)
            elif item_type == "image":
                self._render_image(ax, fig, name, data)
            elif item_type == "histogram":
                self._render_histogram(ax, name, data)

            plt.tight_layout()
            metric_path = f"{base}_{name}{ext}"
            self._output(plt, fig, metric_path, output)

    def _plot_overlaid(self, plt, items, path, output):
        """Plot scalars overlaid; images and histograms in separate subplots."""
        scalars = [(n, d) for t, n, d in items if t == "scalar"]
        images = [(n, d) for t, n, d in items if t == "image"]
        histograms = [(n, d) for t, n, d in items if t == "histogram"]

        if scalars:
            fig, ax = plt.subplots(figsize=(8, 5))
            smoothing = self._get_config("smoothing")

            for name, data in scalars:
                steps, values = zip(*data)
                if smoothing > 0:
                    line, = ax.plot(steps, self._smooth(values), label=name)
                    ax.plot(steps, values, alpha=0.3, color=line.get_color())
                else:
                    ax.plot(steps, values, label=name)

            ax.set_xlabel("step")
            ax.legend()
            plt.tight_layout()
            self._output(plt, fig, path, output)

        if images:
            # Images cannot be overlaid - use subplots layout
            base, ext = os.path.splitext(path)
            if not ext:
                ext = ".png"
            image_path = f"{base}_images{ext}"

            n = len(images)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

            for idx, (name, data) in enumerate(images):
                row, col = divmod(idx, cols)
                ax = axes[row][col]
                self._render_image(ax, fig, name, data)

            for idx in range(n, rows * cols):
                row, col = divmod(idx, cols)
                axes[row][col].axis("off")

            plt.tight_layout()
            self._output(plt, fig, image_path, output)

        if histograms:
            # Histograms cannot be overlaid - use subplots layout
            base, ext = os.path.splitext(path)
            if not ext:
                ext = ".png"
            hist_path = f"{base}_histograms{ext}"

            n = len(histograms)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

            for idx, (name, data) in enumerate(histograms):
                row, col = divmod(idx, cols)
                ax = axes[row][col]
                self._render_histogram(ax, name, data)

            for idx in range(n, rows * cols):
                row, col = divmod(idx, cols)
                axes[row][col].axis("off")

            plt.tight_layout()
            self._output(plt, fig, hist_path, output)

    def _output(self, plt, fig, path, output):
        """Handle output based on configuration."""
        if output in ("file", "both"):
            dpi = self._get_config("dpi")
            fig.savefig(path, dpi=dpi)
            _logger.info(f"QuickLog: saved {path}")
        if output in ("show", "both"):
            plt.show()
        plt.close(fig)
