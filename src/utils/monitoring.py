"""Background resource monitoring thread for benchmark scripts."""

import time
import threading
import psutil

def resource_monitor(
    records: list,
    stop_event: threading.Event,
    interval: float = 0.5,
) -> None:
    """Append CPU% and RAM (MB) snapshots to `records` until stop_event is set."""
    proc = psutil.Process()
    proc.cpu_percent()  # first call always returns 0.0, discard it
    t0 = time.time()

    while not stop_event.is_set():
        records.append({
            'Elapsed_s': round(time.time() - t0, 2),
            'CPU_pct':   proc.cpu_percent(),
            'RAM_MB':    proc.memory_info().rss / (1024 ** 2),
        })
        time.sleep(interval)
