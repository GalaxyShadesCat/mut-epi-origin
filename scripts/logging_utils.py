from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class ProgressState:
    total: int
    start_time: float


def setup_rich_logging(
    *,
    level: int = logging.INFO,
    logger_name: str = "mut_vs_dnase",
    force: bool = True,
) -> logging.Logger:
    """
    Configure logging for compact console output without colors.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="[%X]",
        force=force,
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m{s:04.1f}s"


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("%s", title)


def log_kv(logger: logging.Logger, key: str, value: str) -> None:
    logger.info("  %-20s %s", f"{key}:", value)


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    logger.info("START %s ...", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("DONE %s (%s)", label, _fmt_s(dt))


def progress_line(
    logger: logging.Logger,
    *,
    i: int,
    total: int,
    run_start: float,
    every: int = 5,
    force: bool = False,
) -> None:
    """
    Emit a meaningful progress line every N items (or when forced).
    Includes elapsed, rate, and ETA.
    """
    if not force and i != 1 and i % every != 0 and i != total:
        return

    elapsed = time.perf_counter() - run_start
    rate = (i / elapsed) if elapsed > 0 else float("nan")
    remaining = ((total - i) / rate) if rate and rate > 0 else float("nan")

    logger.info(
        "  chrom %2d/%d  elapsed=%s  rate=%.3f chr/s  eta=%s",
        i,
        total,
        _fmt_s(elapsed),
        rate,
        _fmt_s(remaining) if remaining == remaining else "NA",
    )


def summarise_run(
    logger: logging.Logger,
    *,
    n_bins_total: int,
    n_mutations_total: int,
    correct_celltypes: Optional[str],
    metric_summaries: List[Tuple[str, Optional[str], float]],
    out_paths: Dict[str, str],
) -> None:
    log_section(logger, "Run summary")
    log_kv(logger, "bins_total", _fmt_int(n_bins_total))
    log_kv(logger, "mutations_total", _fmt_int(n_mutations_total))

    log_kv(logger, "correct_celltypes", correct_celltypes or "NA")
    for label, celltype, score in metric_summaries:
        if celltype:
            score_label = f"{score:.4f}" if score == score else "NA"
            logger.info("  %s pred=%s  score=%s", label, celltype, score_label)
        else:
            logger.info("  %s NA", label)

    log_section(logger, "Outputs")
    for k, v in out_paths.items():
        log_kv(logger, k, v)
