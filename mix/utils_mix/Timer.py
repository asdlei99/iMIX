import time


class Timer:

    @classmethod
    def now(cls) -> float:
        return time.perf_counter()

    @classmethod
    def passed_seconds(cls, start: float, end: float) -> float:
        return end - start
