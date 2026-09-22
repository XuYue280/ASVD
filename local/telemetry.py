"""Timing and resource probes for the baseline runs.

WHY THIS EXISTS
    The baseline result JSONs carried exactly one number about cost --
    `compress_seconds` -- while the ARKS records carry six phase timings plus
    peak host RSS and peak CUDA memory. Any cost comparison between the two was
    therefore impossible. This module closes that gap, and it does so by
    transcribing the ARKS definitions rather than inventing new ones, so the two
    sides measure the same quantity and not merely quantities with the same name.

WHERE EACH DEFINITION COMES FROM
    peak_cpu_rss_mb / peak_cpu_rss_increase_mb
        ARKS final1/ksubspaces_artifacts.py:95-143 (class PeakRSS). A daemon
        thread samples field 1 of /proc/self/statm -- resident pages -- every
        10 ms and keeps the maximum. `increase` is peak minus the value at
        entry. NOT resource.getrusage: ru_maxrss is monotonic for the life of
        the process, so it cannot attribute a peak to a phase.
    peak_cuda_allocated_mb / peak_cuda_reserved_mb
        ARKS final1/bench_inference_minibatch_vq.py:1015 (reset), :1146, :1151.
        torch.cuda.reset_peak_memory_stats() at phase entry, then
        max_memory_{allocated,reserved}() at exit, divided by 1024**2.
        `allocated` is what tensors hold; `reserved` is what the caching
        allocator took from the driver. Both are reported because the gap
        between them is the fragmentation headroom a job actually needs.
    *_seconds
        time.perf_counter() deltas, as ARKS does throughout.

    ARKS measures compression and inference in two SEPARATE scripts
    (bench_compress_minibatch_vq.py and bench_inference_minibatch_vq.py), so its
    peak figures are already phase-scoped. The baselines do both in one process,
    which is why every probe here is a scoped context manager: a single
    process-wide peak would be dominated by whichever phase happened to be
    larger and would not be comparable to either ARKS number.

MEASUREMENT ONLY
    Nothing here touches a weight, a dtype, a device placement or a random
    stream. `reset_peak_memory_stats` resets a counter, not the allocator. An
    instrumented run therefore reproduces an uninstrumented one exactly; the
    only cost is one background thread sampling a procfs file.
"""

from __future__ import annotations

import os
import resource
import threading
import time
from pathlib import Path

import torch


def _current_rss_bytes() -> int:
    """Current resident memory without a third-party dependency.

    Transcribed from ARKS final1/ksubspaces_artifacts.py:132-143.
    """
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        return int(peak * 1024 if peak < 10**10 else peak)


class PeakRSS:
    """Sample process RSS in a background thread.

    Transcribed from ARKS final1/ksubspaces_artifacts.py:95-131.
    """

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start_rss = _current_rss_bytes()
        self.peak_rss = self.start_rss

        def sample():
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, _current_rss_bytes())
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, _current_rss_bytes())

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / 1024**2

    @property
    def increase_mb(self) -> float:
        return max(0, self.peak_rss - self.start_rss) / 1024**2


def _usable_devices() -> list:
    """Device indices whose memory counters actually answer.

    `torch.cuda.is_available()` is not sufficient: a CUDA-built torch on a node
    with no GPU reports True, `device_count()` reports 4, and
    `max_memory_allocated(d)` even answers -- it is `reset_peak_memory_stats(d)`
    that raises "Invalid device argument" (observed on trig-login01). So the
    probe here is that exact call, the strictest of the three, and nothing else
    is trusted. Done once so the hot path stays branch-free.
    """
    try:
        if not torch.cuda.is_available():
            return []
        # Force the context up before probing. Without this, a Phase entered
        # before the first CUDA op (notably "total" and "model_load") probes an
        # uninitialised context, gets [], and silently reports None for peak
        # memory -- which is exactly the headline figure. init() is idempotent
        # and the run needs the context anyway, so the only effect is that the
        # ~1 s of context creation is attributed to the first phase instead of
        # to the first CUDA op inside it.
        torch.cuda.init()
        usable = []
        for d in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(d)
                usable.append(d)
            except Exception:
                pass
        return usable
    except Exception:
        return []


class Phase:
    """Scoped wall-clock + peak-memory probe.

        with Phase("compress") as ph:
            ...
        ph.fields()   -> {"compress_seconds": ..., "compress_peak_cuda_allocated_mb": ...}

    CUDA peaks are reset at entry, so each phase reports its OWN peak rather
    than a running process maximum. When several devices are visible the peak is
    summed across them: a model sharded by device_map="auto" holds a slice on
    each card, and the sum is the figure that decides whether the job fits.
    """

    def __init__(self, name: str, devices=None):
        self.name = name
        self._rss = PeakRSS()
        self._t0 = 0.0
        self.seconds = 0.0
        self.cuda_allocated_mb = None
        self.cuda_reserved_mb = None
        self._explicit = None if devices is None else list(devices)
        self._devices = []

    def __enter__(self):
        # Resolved HERE, not in __init__: a Phase object may be constructed long
        # before it is entered, and device usability is only knowable once the
        # process has a CUDA context.
        self._devices = (_usable_devices() if self._explicit is None
                         else list(self._explicit))
        # A probe must never be able to kill a run. torch.cuda.is_available()
        # can be True on a node with no usable device (observed on a CUDA-built
        # torch running on a login node: reset_peak_memory_stats raised
        # "Invalid device argument"), so every CUDA call here is guarded and a
        # failure simply means no CUDA figures are reported.
        try:
            for d in self._devices:
                torch.cuda.reset_peak_memory_stats(d)
        except Exception:
            self._devices = []
        self._rss.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self._rss.__exit__(*exc)
        if self._devices:
            try:
                self.cuda_allocated_mb = sum(
                    torch.cuda.max_memory_allocated(d) for d in self._devices) / 1024**2
                self.cuda_reserved_mb = sum(
                    torch.cuda.max_memory_reserved(d) for d in self._devices) / 1024**2
            except Exception:
                self.cuda_allocated_mb = self.cuda_reserved_mb = None
        return False          # never swallow an exception

    def fields(self) -> dict:
        p = self.name
        return {
            f"{p}_seconds": self.seconds,
            f"{p}_peak_cuda_allocated_mb": self.cuda_allocated_mb,
            f"{p}_peak_cuda_reserved_mb": self.cuda_reserved_mb,
            f"{p}_peak_cpu_rss_mb": self._rss.peak_mb,
            f"{p}_peak_cpu_rss_increase_mb": self._rss.increase_mb,
        }


def rollup(outer, children) -> dict:
    """Correct an enclosing phase's CUDA peaks using its children's.

    THE BUG THIS FIXES. torch's peak counter is per-device and GLOBAL, and every
    scoped Phase resets it on entry. An enclosing phase therefore cannot read its
    own peak: by the time it exits, the counter only reflects activity since the
    LAST child reset it. Measured on opt-125m: compress peaked at 1657.7 MB and
    eval at 799.1 MB, and the enclosing "total" reported 799.1 -- the last child's
    figure, not the maximum. Reported as a whole-run peak that is simply wrong.

    Host RSS does not have this problem: PeakRSS samples continuously in a thread
    and is unaffected by any child, so the outer phase's RSS figures are already
    correct and are left alone.

    `outer` is the enclosing Phase, `children` the ones nested inside it.
    """
    fields = outer.fields()
    for suffix in ("peak_cuda_allocated_mb", "peak_cuda_reserved_mb"):
        vals = [c.fields().get(f"{c.name}_{suffix}") for c in children]
        vals = [v for v in vals if v is not None]
        own = fields.get(f"{outer.name}_{suffix}")
        if own is not None:
            vals.append(own)
        fields[f"{outer.name}_{suffix}"] = max(vals) if vals else None
    return fields


def environment() -> dict:
    """What the measurements above were taken on.

    A runtime or a peak-memory figure is meaningless without this: the same job
    on a different card, driver or torch build is a different measurement.
    """
    import platform

    info = {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        # The cgroup/affinity-aware count -- on a scheduler this is what the job
        # actually got, which os.cpu_count() does not report.
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }
    _dev = _usable_devices()
    info["gpu_count"] = len(_dev)
    try:
        info["gpu_names"] = [torch.cuda.get_device_name(d) for d in _dev]
        info["gpu_total_memory_mb"] = [
            torch.cuda.get_device_properties(d).total_memory / 1024**2 for d in _dev]
    except Exception:
        info["gpu_names"] = []
        info["gpu_total_memory_mb"] = []
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    # Present only under Slurm; harmless (and absent) on a local box.
    for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK"):
        if os.environ.get(k):
            info[k.lower()] = os.environ[k]
    return info
