"""Tests for monitoring modules — thermal, memory, recorder."""

import time
import pytest
import numpy as np
from tinytpu.monitoring.thermal import ThermalMonitor
from tinytpu.monitoring.memory import MemoryWatchdog
from tinytpu.monitoring.recorder import BlackBoxRecorder


class TestThermalMonitor:

    def test_creation(self):
        tm = ThermalMonitor()
        assert tm is not None

    def test_get_temp_returns_float(self):
        tm = ThermalMonitor()
        temp = tm.get_temp()
        assert isinstance(temp, (int, float))
        assert temp >= 0.0  # even on Windows (returns 0.0)

    def test_start_stop(self):
        tm = ThermalMonitor()
        tm.start()
        time.sleep(0.1)
        tm.stop()

    def test_peak_temp(self):
        tm = ThermalMonitor()
        _ = tm.get_temp()
        assert tm.peak_temp >= 0.0

    def test_initial_state(self):
        tm = ThermalMonitor()
        assert tm.state == "ok"

    def test_warn_threshold_configurable(self):
        tm = ThermalMonitor(warn_temp=50.0, critical_temp=60.0)
        assert tm.warn_temp == 50.0
        assert tm.critical_temp == 60.0

    def test_callbacks_stored(self):
        called = []
        tm = ThermalMonitor(on_warning=lambda: called.append("warn"))
        assert tm.on_warning is not None


class TestMemoryWatchdog:

    def test_creation(self):
        mw = MemoryWatchdog()
        assert mw is not None

    def test_get_rss_returns_nonnegative(self):
        mw = MemoryWatchdog()
        rss = mw._get_rss_mb()
        assert rss >= 0.0

    def test_peak_rss(self):
        mw = MemoryWatchdog()
        mw.current_rss_mb = 100.0
        mw.peak_rss_mb = max(mw.peak_rss_mb, mw.current_rss_mb)
        assert mw.peak_rss_mb >= 100.0

    def test_start_stop(self):
        mw = MemoryWatchdog()
        mw.start()
        time.sleep(0.1)
        mw.stop()

    def test_initial_state(self):
        mw = MemoryWatchdog()
        assert mw.state == "ok"

    def test_get_status(self):
        mw = MemoryWatchdog()
        status = mw.get_status()
        assert "state" in status
        assert "current_rss_mb" in status
        assert "peak_rss_mb" in status
        assert "max_rss_mb" in status

    def test_configurable(self):
        mw = MemoryWatchdog(max_rss_mb=1024, warn_percent=70)
        assert mw.max_rss_mb == 1024
        assert mw.warn_percent == 70


class TestBlackBoxRecorder:

    def test_creation(self):
        r = BlackBoxRecorder()
        assert r is not None

    def test_record_event(self):
        r = BlackBoxRecorder()
        r.record({"type": "test_event", "key": "value"})
        events = r.last(1)
        assert len(events) == 1
        assert events[0]["type"] == "test_event"
        assert events[0]["key"] == "value"

    def test_record_multiple(self):
        r = BlackBoxRecorder()
        for i in range(10):
            r.record({"type": "event", "i": i})
        events = r.last(5)
        assert len(events) == 5

    def test_circular_buffer_evicts(self):
        r = BlackBoxRecorder(max_events=5)
        for i in range(10):
            r.record({"type": "event", "i": i})
        events = r.last(10)
        assert len(events) == 5
        # Should have the last 5 events
        assert events[-1]["i"] == 9

    def test_save_load(self, tmp_path):
        r = BlackBoxRecorder()
        r.record({"type": "detection", "class": "person", "conf": 0.9})
        r.record({"type": "command", "linear": 0.2, "angular": 0.1})
        path = str(tmp_path / "blackbox.json")
        r.save(path)

        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 2

    def test_last_empty(self):
        r = BlackBoxRecorder()
        events = r.last(5)
        assert events == []

    def test_event_has_timestamp(self):
        r = BlackBoxRecorder()
        r.record({"type": "test"})
        event = r.last(1)[0]
        assert "t" in event  # monotonic timestamp
        assert isinstance(event["t"], (int, float))
