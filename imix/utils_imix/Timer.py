import time


class Timer:

    @classmethod
    def now(cls) -> float:
        return time.perf_counter()

    @classmethod
    def passed_seconds(cls, start: float, end: float) -> float:
        return end - start

    @classmethod
    def run_time(cls, func):

        def wrapper(*args, **kwargs):
            start = Timer.now()
            result = func(*args, **kwargs)
            passed_time = Timer.passed_seconds(start=start, end=Timer.now())
            return result, passed_time

        return wrapper
