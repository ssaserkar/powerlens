
"""
Background power sampler.

Runs in a separate thread, reading power sensors at a configurable
rate and storing timestamped samples in a thread-safe buffer.
"""

import threading
import time
import logging
from typing import List, Optional

from powerlens.sensors.types import PowerSample

logger = logging.getLogger(__name__)


class PowerSampler:
    """
    Background thread that continuously samples power sensors.

    Usage:
        sampler = PowerSampler(sensor, sample_rate_hz=100)
        sampler.start()
        # ... run your workload ...
        sampler.stop()
        samples = sampler.get_samples()
    """

    def __init__(self, sensor, sample_rate_hz: float = 100.0):
        self._sensor = sensor
        self._sample_rate_hz = sample_rate_hz
        self._samples: List[List[PowerSample]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._sample_count = 0

    @property
    def sample_interval(self) -> float:
        """Target time between samples in seconds."""
        return 1.0 / self._sample_rate_hz

    def start(self):
        """Start background sampling."""
        if self._running:
            raise RuntimeError("Sampler is already running")

        self._samples = []
        self._sample_count = 0
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
            name="powerlens-sampler",
        )
        self._thread.start()
        logger.info(
            "Power sampler started at %.1f Hz target",
            self._sample_rate_hz,
        )

    def stop(self):
        """Stop background sampling and wait for thread to finish."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        actual_rate = self._sample_count / elapsed if elapsed > 0 else 0
        logger.info(
            "Power sampler stopped. %d samples in %.2fs (%.1f Hz actual)",
            self._sample_count,
            elapsed,
            actual_rate,
        )

    def _sample_loop(self):
        """Main sampling loop running in background thread."""
        interval = self.sample_interval

        while self._running:
            loop_start = time.monotonic()

            try:
                channel_samples = self._sensor.read_all()
                with self._lock:
                    self._samples.append(channel_samples)
                    self._sample_count += 1
            except Exception as e:
                logger.warning("Sensor read failed: %s", e)

            # Sleep for remaining interval time
            elapsed = time.monotonic() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_samples(self) -> List[List[PowerSample]]:
        """Return a copy of all collected samples.

        Each element is a list of PowerSample (one per channel)
        from a single read cycle.
        """
        with self._lock:
            return list(self._samples)

    @property
    def sample_count(self) -> int:
        """Number of sample cycles collected so far."""
        with self._lock:
            return self._sample_count

    @property
    def is_running(self) -> bool:
        """Whether the sampler is currently running."""
        return self._running
