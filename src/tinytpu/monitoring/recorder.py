"""TinyTPU Black Box Recorder."""
import time, json, logging
from collections import deque

logger = logging.getLogger("tinytpu.monitoring.recorder")


class BlackBoxRecorder:
    def __init__(self, max_events=10000):
        self.events = deque(maxlen=max_events)
        self._start = time.monotonic()

    def record(self, data: dict):
        self.events.append({"t": round(time.monotonic() - self._start, 3), **data})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(list(self.events), f)

    def last(self, n: int = 10):
        return list(self.events)[-n:]
