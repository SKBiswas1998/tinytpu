"""TinyTPU Memory Watchdog."""
import os, time, threading, gc, logging

logger = logging.getLogger("tinytpu.monitoring.memory")


class MemoryWatchdog:
    def __init__(self, max_rss_mb=512, warn_percent=80, poll_interval=5.0,
                 on_warning=None, on_critical=None):
        self.max_rss_mb = max_rss_mb
        self.warn_percent = warn_percent
        self.poll_interval = poll_interval
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.current_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.state = "ok"
        self._running = False
        self._thread = None

    def _get_rss_mb(self) -> float:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            pass
        try:
            with open(f"/proc/{os.getpid()}/statm") as f:
                pages = int(f.read().split()[1])
                return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
        except (OSError, ValueError):
            return 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True, name="mem_watchdog")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while self._running:
            self.current_rss_mb = self._get_rss_mb()
            self.peak_rss_mb = max(self.peak_rss_mb, self.current_rss_mb)
            if self.current_rss_mb > self.max_rss_mb:
                self.state = "critical"
                gc.collect()
                if self.on_critical:
                    self.on_critical()
            elif self.current_rss_mb > self.max_rss_mb * self.warn_percent / 100:
                self.state = "warn"
                if self.on_warning:
                    self.on_warning()
            else:
                self.state = "ok"
            time.sleep(self.poll_interval)

    def get_status(self) -> dict:
        return {
            "state": self.state, "current_rss_mb": round(self.current_rss_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1), "max_rss_mb": self.max_rss_mb,
        }
