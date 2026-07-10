import logging
import os
from multiprocessing import Lock, Value
from typing import Optional


class ConcurrencyException(Exception):
    pass


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1) -> None:
        self.max_concurrency = max_concurrency
        self.lock = Lock()
        # Use lock=False: all writes are protected by self.lock, and reads of
        # a single aligned int from shared memory are atomic on x86/ARM.
        # Removing the implicit per-Value lock avoids redundant syscall overhead,
        # especially on the fast-path where most requests are rejected without
        # acquiring self.lock.
        self.current_concurrency = Value("i", 0, lock=False)
        self.request_counter = Value("i", 0, lock=False)

    def get_available_concurrency(self) -> int:
        with self.lock:
            return self.max_concurrency - self.current_concurrency.value

    def increment(self) -> None:
        # Fast path: lock-free observation.  When the limit is already reached,
        # reject immediately without acquiring the cross-process lock.  This
        # prevents the event loop from being starved by lock contention under
        # high-volume 503 storms.
        if self.current_concurrency.value >= self.max_concurrency:
            raise ConcurrencyException(
                f"Concurrency limit {self.max_concurrency} reached"
            )

        # Slow path: capacity appears available — acquire lock and double-check
        # (TOCTOU guard: value may have changed between observation and lock).
        with self.lock:
            if self.current_concurrency.value < self.max_concurrency:
                self.current_concurrency.value += 1
                self.request_counter.value += 1
                return self.request_counter.value

            raise ConcurrencyException(
                f"Concurrency limit {self.max_concurrency} reached"
            )

    def decrement(self) -> None:
        with self.lock:
            self.current_concurrency.value -= 1

    def get_request_counter(self) -> int:
        with self.lock:
            return self.request_counter.value

    def __enter__(self) -> None:
        self.increment()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.decrement()


global_controller: Optional[ConcurrencyController] = None


def init_controller(concurrency_config, dp_size=1):
    """Initialize concurrency controller.

    Args:
        concurrency_config: ConcurrencyConfig object.
        dp_size: Data parallel size. If None,
    """

    concurrency_limit = concurrency_config.concurrency_limit
    global_concurrency_limit = concurrency_limit * dp_size
    logging.info(
        f"concurrency_limit : {concurrency_limit}, global_concurrency_limit : {global_concurrency_limit}"
    )
    controller = ConcurrencyController(global_concurrency_limit)
    return controller


def set_global_controller(_global_controller: ConcurrencyController):
    global global_controller
    global_controller = _global_controller


def get_global_controller() -> Optional[ConcurrencyController]:
    return global_controller
