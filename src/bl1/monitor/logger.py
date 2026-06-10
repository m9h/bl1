"""Unified experiment logger for BL-1 + doom-neuron.

Writes to both trackio (JSON, Gradio dashboard) and TensorBoard
(TFRecord, TensorBoard web UI) simultaneously.  Either backend
can be disabled explicitly or will be skipped automatically if its
dependency is not installed.

Backends
--------
* **trackio** -- JSON-based experiment tracking with a Gradio dashboard.
  Used by the BL-1 training loop (``scripts/train_all_sharf.py``,
  ``src/bl1/training/trainer.py``).
* **TensorBoard** -- Binary TFRecord logs viewed via the TensorBoard web
  UI.  Used by doom-neuron (``ppo_doom.py``, ``training_server.py``).

Usage
-----
::

    from bl1.monitor.logger import UnifiedLogger

    logger = UnifiedLogger(
        project="bl1-doom",
        run_name="run_001",
        log_dir="./logs",
        config={"n_neurons": 10000, "tick_hz": 10},
    )

    # Log scalars (writes to both backends)
    logger.log_scalar("Reward/episode", 42.5, step=100)
    logger.log_scalars({"train/loss": 0.5, "train/fr_hz": 1.2}, step=100)

    # Log config (trackio config dict + TensorBoard hparams)
    logger.log_config({"n_neurons": 10000, "learning_rate": 3e-4})

    # Log image (TensorBoard only -- trackio does not support images)
    logger.log_image("Neural/mea_heatmap", image_array, step=100)

    # Log histogram (TensorBoard only)
    logger.log_histogram("Spikes/distribution", spike_counts, step=100)

    # Flush and close
    logger.close()

Drop-in replacement for doom-neuron
------------------------------------
doom-neuron code uses ``torch.utils.tensorboard.SummaryWriter`` extensively.
:class:`TensorBoardAdapter` is a drop-in replacement that also logs to
trackio:

::

    # Old:
    # from torch.utils.tensorboard import SummaryWriter
    # self.writer = SummaryWriter(config.log_dir)

    # New:
    from bl1.monitor.logger import TensorBoardAdapter as SummaryWriter
    self.writer = SummaryWriter(config.log_dir, project="doom-neuron")

Drop-in replacement for BL-1 trainer
-------------------------------------
BL-1's ``train_weights()`` accepts a ``tracker`` object with a ``.log(dict)``
method.  :class:`UnifiedLogger` satisfies that interface directly:

::

    # Old:
    # run_tracker = trackio.init(project="bl1-sharf", name=run_name, ...)
    # trackio.log({"train/loss": loss_val})

    # New:
    from bl1.monitor.logger import UnifiedLogger
    logger = UnifiedLogger(
        project="bl1-sharf", run_name=run_name, log_dir=output_dir,
    )
    # Pass logger as the tracker -- it has a .log(dict) method
    result = train_weights(config, tracker=logger)
"""

from __future__ import annotations

import contextlib
import logging
import threading
import warnings
from pathlib import Path
from typing import Any, Union

__all__ = ["UnifiedLogger", "TensorBoardAdapter"]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy backend imports -- never crash if a dependency is missing
# ---------------------------------------------------------------------------

_HAS_TRACKIO = False
_HAS_TENSORBOARD = False

try:
    import trackio as _trackio  # noqa: F401

    _HAS_TRACKIO = True
except ImportError:
    _trackio = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # noqa: F401

    _HAS_TENSORBOARD = True
except ImportError:
    _SummaryWriter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Key normalisation helpers
# ---------------------------------------------------------------------------


def _to_tb_tag(key: str) -> str:
    """Convert a dot-delimited trackio key to a ``/``-delimited TensorBoard tag.

    Examples
    --------
    >>> _to_tb_tag("train.loss")
    'train/loss'
    >>> _to_tb_tag("Reward/episode")
    'Reward/episode'
    """
    return key.replace(".", "/")


def _to_trackio_key(key: str) -> str:
    """Convert a ``/``-delimited TensorBoard tag to a dot-delimited trackio key.

    Examples
    --------
    >>> _to_trackio_key("Reward/episode")
    'Reward.episode'
    >>> _to_trackio_key("train.loss")
    'train.loss'
    """
    return key.replace("/", ".")


# ---------------------------------------------------------------------------
# UnifiedLogger
# ---------------------------------------------------------------------------


class UnifiedLogger:
    """Dual-backend experiment logger (trackio + TensorBoard).

    Parameters
    ----------
    project : str
        Project name (used by trackio and as a TensorBoard sub-directory).
    run_name : str or None
        Human-readable run identifier.  Defaults to ``None`` which lets
        each backend pick its own default name.
    log_dir : str or Path
        Root directory for log files.  TensorBoard events are written under
        ``<log_dir>/tensorboard/<run_name>/``.  trackio writes its own
        files under ``<log_dir>/trackio/``.
    config : dict or None
        Hyperparameter dict logged at init time to both backends.
    use_trackio : bool
        Enable the trackio backend (still requires the package to be
        installed).
    use_tensorboard : bool
        Enable the TensorBoard backend (still requires
        ``torch.utils.tensorboard`` to be installed).

    Notes
    -----
    * All public methods are **thread-safe** -- an internal lock serialises
      writes so the logger can be shared across training and simulation
      threads.
    * If a requested backend is not installed the logger emits a single
      warning and continues with whatever backends *are* available.
    * The instance exposes a ``.log(dict)`` method so it can be passed
      directly to ``train_weights(config, tracker=logger)`` in BL-1.
    """

    def __init__(
        self,
        project: str = "experiment",
        run_name: str | None = None,
        log_dir: Union[str, Path] = "./logs",
        config: dict[str, Any] | None = None,
        use_trackio: bool = True,
        use_tensorboard: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._step: int = 0
        self._closed: bool = False

        self.project = project
        self.run_name = run_name
        self.log_dir = Path(log_dir)

        # ---- trackio --------------------------------------------------
        self._trackio_run: Any = None
        if use_trackio:
            if not _HAS_TRACKIO:
                warnings.warn(
                    "trackio is not installed -- trackio backend disabled. "
                    "Install with: pip install trackio",
                    stacklevel=2,
                )
            else:
                try:
                    trackio_dir = self.log_dir / "trackio"
                    trackio_dir.mkdir(parents=True, exist_ok=True)
                    self._trackio_run = _trackio.init(
                        project=project,
                        name=run_name,
                        dir=str(trackio_dir),
                        config=config or {},
                    )
                    _log.info("trackio backend initialised (project=%s)", project)
                except Exception:
                    _log.exception("Failed to initialise trackio backend")
                    self._trackio_run = None

        # ---- TensorBoard ---------------------------------------------
        self._tb_writer: Any = None
        if use_tensorboard:
            if not _HAS_TENSORBOARD:
                warnings.warn(
                    "torch.utils.tensorboard is not installed -- TensorBoard "
                    "backend disabled. Install with: pip install tensorboard",
                    stacklevel=2,
                )
            else:
                try:
                    tb_dir = self.log_dir / "tensorboard"
                    if run_name:
                        tb_dir = tb_dir / run_name
                    tb_dir.mkdir(parents=True, exist_ok=True)
                    self._tb_writer = _SummaryWriter(str(tb_dir))
                    _log.info("TensorBoard backend initialised (log_dir=%s)", tb_dir)
                except Exception:
                    _log.exception("Failed to initialise TensorBoard backend")
                    self._tb_writer = None

        # ---- Log initial config if provided ---------------------------
        if config:
            self.log_config(config)

        if self._trackio_run is None and self._tb_writer is None:
            warnings.warn(
                "UnifiedLogger: no backends available -- all logging will be "
                "no-ops.  Install trackio and/or tensorboard.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current auto-increment step counter."""
        return self._step

    @property
    def has_trackio(self) -> bool:
        """Whether the trackio backend is active."""
        return self._trackio_run is not None

    @property
    def has_tensorboard(self) -> bool:
        """Whether the TensorBoard backend is active."""
        return self._tb_writer is not None

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def _resolve_step(self, step: int | None) -> int:
        """Return *step* if given, otherwise auto-increment."""
        if step is not None:
            self._step = max(self._step, step)
            return step
        self._step += 1
        return self._step

    def log_scalar(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a single scalar metric to both backends.

        Parameters
        ----------
        key : str
            Metric name.  Slash-delimited (``Loss/policy``) and
            dot-delimited (``train.loss``) forms are both accepted;
            each backend receives the appropriately-normalised key.
        value : float
            Scalar value.
        step : int or None
            Global step.  If *None* the internal counter is
            auto-incremented.
        """
        with self._lock:
            if self._closed:
                return
            s = self._resolve_step(step)

            if self._trackio_run is not None:
                try:
                    self._trackio_run.log({_to_trackio_key(key): float(value)})
                except Exception:
                    _log.exception("trackio.log failed for key=%s", key)

            if self._tb_writer is not None:
                try:
                    self._tb_writer.add_scalar(_to_tb_tag(key), float(value), s)
                except Exception:
                    _log.exception("TensorBoard add_scalar failed for key=%s", key)

    def log_scalars(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a dict of scalar metrics to both backends.

        This is more efficient than calling :meth:`log_scalar` in a loop
        because the trackio backend receives a single ``log()`` call with
        the full dict.

        Parameters
        ----------
        metrics : dict
            ``{key: value}`` mapping of scalar metrics.
        step : int or None
            Global step.  If *None* the internal counter is
            auto-incremented once for the entire batch.
        """
        if not metrics:
            return
        with self._lock:
            if self._closed:
                return
            s = self._resolve_step(step)

            if self._trackio_run is not None:
                try:
                    trackio_dict = {_to_trackio_key(k): float(v) for k, v in metrics.items()}
                    self._trackio_run.log(trackio_dict)
                except Exception:
                    _log.exception("trackio.log failed for batch of %d metrics", len(metrics))

            if self._tb_writer is not None:
                for k, v in metrics.items():
                    try:
                        self._tb_writer.add_scalar(_to_tb_tag(k), float(v), s)
                    except Exception:
                        _log.exception("TensorBoard add_scalar failed for key=%s", k)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of metrics -- trackio-compatible interface.

        This method exists so that a :class:`UnifiedLogger` instance can
        be passed directly as the ``tracker`` argument to
        ``bl1.training.trainer.train_weights()``, which calls
        ``tracker.log({...})``.

        Parameters
        ----------
        metrics : dict
            ``{key: value}`` mapping of scalar metrics.
        step : int or None
            Global step.  If *None* the internal counter is
            auto-incremented.
        """
        self.log_scalars(metrics, step=step)

    # ------------------------------------------------------------------
    # Config / hyperparameters
    # ------------------------------------------------------------------

    def log_config(self, config: dict[str, Any]) -> None:
        """Log hyperparameters / configuration to both backends.

        Parameters
        ----------
        config : dict
            Flat dict of hyperparameters (values should be scalars or
            strings).

        Notes
        -----
        * trackio receives the config via ``run.log({"config/...": ...})``
          (the initial config dict is already passed at init time).
        * TensorBoard receives the config via ``add_hparams()``.  Since
          ``add_hparams`` requires a metric dict, a placeholder
          ``hparam/placeholder`` metric is used.
        """
        with self._lock:
            if self._closed:
                return

            if self._trackio_run is not None:
                try:
                    trackio_cfg = {
                        f"config.{_to_trackio_key(k)}": _coerce_for_trackio(v)
                        for k, v in config.items()
                    }
                    self._trackio_run.log(trackio_cfg)
                except Exception:
                    _log.exception("trackio config logging failed")

            if self._tb_writer is not None:
                try:
                    hparam_dict = {k: _coerce_for_hparams(v) for k, v in config.items()}
                    # add_hparams requires at least one metric
                    self._tb_writer.add_hparams(
                        hparam_dict,
                        {"hparam/placeholder": 0.0},
                    )
                except Exception:
                    _log.exception("TensorBoard add_hparams failed")

    # ------------------------------------------------------------------
    # TensorBoard-only logging (images, histograms, text)
    # ------------------------------------------------------------------

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an image to TensorBoard.

        Parameters
        ----------
        tag : str
            Image tag.
        img_tensor : array-like
            Image data.  Accepted shapes: ``(H, W)``, ``(H, W, C)``,
            ``(C, H, W)``.  See TensorBoard ``add_image`` docs.
        step : int or None
            Global step.
        **kwargs
            Forwarded to ``SummaryWriter.add_image()``.

        Notes
        -----
        trackio does not support image logging -- this method is a
        TensorBoard-only operation.
        """
        with self._lock:
            if self._closed:
                return
            s = self._resolve_step(step)

            if self._tb_writer is not None:
                try:
                    self._tb_writer.add_image(_to_tb_tag(tag), img_tensor, s, **kwargs)
                except Exception:
                    _log.exception("TensorBoard add_image failed for tag=%s", tag)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a histogram to TensorBoard.

        Parameters
        ----------
        tag : str
            Histogram tag.
        values : array-like
            Values for the histogram.
        step : int or None
            Global step.
        **kwargs
            Forwarded to ``SummaryWriter.add_histogram()``.

        Notes
        -----
        trackio does not support histogram logging -- this method is a
        TensorBoard-only operation.
        """
        with self._lock:
            if self._closed:
                return
            s = self._resolve_step(step)

            if self._tb_writer is not None:
                try:
                    self._tb_writer.add_histogram(
                        _to_tb_tag(tag),
                        values,
                        s,
                        **kwargs,
                    )
                except Exception:
                    _log.exception("TensorBoard add_histogram failed for tag=%s", tag)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int | None = None,
    ) -> None:
        """Log text to TensorBoard.

        Parameters
        ----------
        tag : str
            Text tag.
        text : str
            The text string to log.
        step : int or None
            Global step.

        Notes
        -----
        trackio does not support text logging -- this method is a
        TensorBoard-only operation.
        """
        with self._lock:
            if self._closed:
                return
            s = self._resolve_step(step)

            if self._tb_writer is not None:
                try:
                    self._tb_writer.add_text(_to_tb_tag(tag), text, s)
                except Exception:
                    _log.exception("TensorBoard add_text failed for tag=%s", tag)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush pending writes on all backends."""
        with self._lock:
            if self._tb_writer is not None:
                try:
                    self._tb_writer.flush()
                except Exception:
                    _log.exception("TensorBoard flush failed")

    def close(self) -> None:
        """Flush and close all backends.

        After calling ``close()`` all subsequent log calls are silently
        ignored.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._tb_writer is not None:
                try:
                    self._tb_writer.close()
                except Exception:
                    _log.exception("TensorBoard close failed")
                self._tb_writer = None

            # trackio runs do not expose a close/finish method -- the
            # run object is garbage-collected and the background writer
            # flushes on process exit.
            self._trackio_run = None

    def __enter__(self) -> UnifiedLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()


# ---------------------------------------------------------------------------
# TensorBoardAdapter -- drop-in replacement for SummaryWriter
# ---------------------------------------------------------------------------


class TensorBoardAdapter:
    """Drop-in replacement for ``torch.utils.tensorboard.SummaryWriter``
    that also logs to trackio.

    This adapter mirrors the ``SummaryWriter`` interface so that existing
    doom-neuron code can switch backends with a single import change:

    ::

        # Old:
        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(config.log_dir)

        # New:
        from bl1.monitor.logger import TensorBoardAdapter as SummaryWriter
        self.writer = SummaryWriter(config.log_dir, project="doom-neuron")

    All calls are forwarded to an internal :class:`UnifiedLogger`.

    Parameters
    ----------
    log_dir : str or Path
        Directory for log files (matches ``SummaryWriter.__init__``
        signature).
    project : str
        Project name for trackio.
    run_name : str or None
        Run identifier.
    config : dict or None
        Hyperparameters to log at init time.
    use_trackio : bool
        Enable the trackio backend.
    use_tensorboard : bool
        Enable the TensorBoard backend.
    purge_step : int or None
        If set, passed to the TensorBoard backend as the ``purge_step``
        parameter (used when resuming a training run to discard stale
        events after this step).
    **kwargs
        Silently ignored so that callers passing extra ``SummaryWriter``
        keyword arguments (e.g. ``comment``, ``flush_secs``) do not
        crash.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "./logs",
        project: str = "experiment",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        use_trackio: bool = True,
        use_tensorboard: bool = True,
        purge_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._logger = UnifiedLogger(
            project=project,
            run_name=run_name,
            log_dir=log_dir,
            config=config,
            use_trackio=use_trackio,
            use_tensorboard=use_tensorboard,
        )
        # Expose the underlying TensorBoard log_dir for code that reads
        # writer.log_dir (doom-neuron does this).
        self.log_dir: str = str(Path(log_dir))

        # Apply purge_step if the underlying TB writer supports it
        if purge_step is not None and self._logger._tb_writer is not None:
            try:
                # Re-create the writer with purge_step
                tb_dir = self._logger._tb_writer.log_dir
                self._logger._tb_writer.close()
                self._logger._tb_writer = _SummaryWriter(tb_dir, purge_step=purge_step)
            except Exception:
                _log.exception("Failed to apply purge_step=%d", purge_step)

    # ---- SummaryWriter-compatible methods ----------------------------

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a scalar value (mirrors ``SummaryWriter.add_scalar``)."""
        self._logger.log_scalar(tag, scalar_value, step=global_step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log multiple scalars under a group tag (mirrors ``SummaryWriter.add_scalars``)."""
        merged = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self._logger.log_scalars(merged, step=global_step)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an image (mirrors ``SummaryWriter.add_image``)."""
        self._logger.log_image(tag, img_tensor, step=global_step, **kwargs)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a histogram (mirrors ``SummaryWriter.add_histogram``)."""
        self._logger.log_histogram(tag, values, step=global_step, **kwargs)

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log text (mirrors ``SummaryWriter.add_text``)."""
        self._logger.log_text(tag, text_string, step=global_step)

    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log hyperparameters (mirrors ``SummaryWriter.add_hparams``)."""
        self._logger.log_config(hparam_dict)
        if metric_dict:
            self._logger.log_scalars(metric_dict)

    def flush(self) -> None:
        """Flush pending writes."""
        self._logger.flush()

    def close(self) -> None:
        """Close all backends."""
        self._logger.close()

    def __enter__(self) -> TensorBoardAdapter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_for_trackio(value: Any) -> Any:
    """Coerce a config value to a type trackio can serialise (JSON-safe)."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    # numpy scalars, JAX arrays, torch tensors -- extract Python scalar
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _coerce_for_hparams(value: Any) -> Any:
    """Coerce a config value to a type TensorBoard ``add_hparams`` accepts.

    TensorBoard hparams only accepts ``int | float | str | bool | Tensor``.
    """
    if isinstance(value, (int, float, str, bool)):
        return value
    if value is None:
        return "None"
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
