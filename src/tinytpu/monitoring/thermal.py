"""TinyTPU Thermal Monitor."""
import time
import threading
import logging

logger = logging.getLogger("tinytpu.monitoring.thermal")


class ThermalMonitor:
    def __init__(self, warn_temp=70.0, critical_temp=80.0, poll_interval=5.0,
                 on_warning=None, on_critical=None):
        self.warn_temp = warn_temp
        self.critical_temp = critical_temp
        self.poll_interval = poll_interval
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.current_temp = 0.0
        self.peak_temp = 0.0
        self.state = "ok"
        self._running = False
        self._thread = None

    def get_temp(self) -> float:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True, name="thermal")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while self._running:
            self.current_temp = self.get_temp()
            self.peak_temp = max(self.peak_temp, self.current_temp)
            if self.current_temp >= self.critical_temp:
                self.state = "critical"
                if self.on_critical:
                    self.on_critical()
            elif self.current_temp >= self.warn_temp:
                self.state = "warn"
                if self.on_warning:
                    self.on_warning()
            else:
                self.state = "ok"
            time.sleep(self.poll_interval)
