"""
Simple metrics collector for SnapDB.
Tracks QPS, latency histograms, and basic counters.
Zero-dependency, thread-safe using atomic operations (not fully thread-safe but okay for GIL).
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict


class Metrics:
    """
    Collects metrics for SnapDB operations.
    Usage:
        metrics = Metrics()
        db = SnapDB(..., metrics=metrics)
        # later:
        print(metrics.report())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._last_report_time = self._start_time
        self._last_report_counts: Dict[str, int] = defaultdict(int)

    def _now(self) -> float:
        return time.time()

    def inc(self, metric: str, value: int = 1) -> None:
        with self._lock:
            self._counters[metric] += value

    def add_latency(self, metric: str, seconds: float) -> None:
        with self._lock:
            self._latencies[metric].append(seconds)

    def _snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def _latency_snapshot(self) -> Dict[str, List[float]]:
        with self._lock:
            return {k: v.copy() for k, v in self._latencies.items()}

    def report(self) -> str:
        """Return a Prometheus-like text report."""
        lines = []
        uptime = self._now() - self._start_time
        lines.append(f"# HELP snapdb_uptime_seconds Seconds since metrics started")
        lines.append(f"# TYPE snapdb_uptime_seconds gauge")
        lines.append(f"snapdb_uptime_seconds {uptime:.3f}")

        with self._lock:
            for metric, count in self._counters.items():
                lines.append(f"# HELP snapdb_{metric}_total Total number of {metric}")
                lines.append(f"# TYPE snapdb_{metric}_total counter")
                lines.append(f"snapdb_{metric}_total {count}")

            for metric, values in self._latencies.items():
                if not values:
                    continue
                # compute percentiles
                sorted_vals = sorted(values)
                count = len(sorted_vals)
                def pct(p):
                    idx = int(count * p / 100)
                    if idx >= count:
                        idx = count - 1
                    return sorted_vals[idx]
                p50 = pct(50)
                p95 = pct(95)
                p99 = pct(99)
                avg = sum(sorted_vals) / count
                lines.append(f"# HELP snapdb_{metric}_latency_seconds Latency of {metric}")
                lines.append(f"# TYPE snapdb_{metric}_latency_seconds summary")
                lines.append(f"snapdb_{metric}_latency_seconds{{quantile=\"0.5\"}} {p50:.6f}")
                lines.append(f"snapdb_{metric}_latency_seconds{{quantile=\"0.95\"}} {p95:.6f}")
                lines.append(f"snapdb_{metric}_latency_seconds{{quantile=\"0.99\"}} {p99:.6f}")
                lines.append(f"snapdb_{metric}_latency_seconds_sum {sum(sorted_vals):.6f}")
                lines.append(f"snapdb_{metric}_latency_seconds_count {count}")

        # QPS since last report
        now = self._now()
        elapsed = now - self._last_report_time
        if elapsed > 0:
            with self._lock:
                current = dict(self._counters)
                for metric, count in current.items():
                    last = self._last_report_counts.get(metric, 0)
                    qps = (count - last) / elapsed if elapsed > 0 else 0.0
                    lines.append(f"# HELP snapdb_{metric}_per_second Per second rate of {metric}")
                    lines.append(f"# TYPE snapdb_{metric}_per_second gauge")
                    lines.append(f"snapdb_{metric}_per_second {qps:.2f}")
                self._last_report_time = now
                self._last_report_counts = current

        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._start_time = self._now()
            self._last_report_time = self._start_time
            self._last_report_counts.clear()


# Decorator to time a method and record latency
def timeit(metrics: Optional[Metrics], metric_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                if metrics is not None:
                    metrics.add_latency(metric_name, time.time() - start)
        return wrapper
    return decorator


if __name__ == "__main__":
    m = Metrics()
    m.inc("inserts", 10)
    m.add_latency("inserts", 0.001)
    m.add_latency("inserts", 0.002)
    print(m.report())
