from typing import Type, Optional


# INFO need to use string references to avoid circular imports and deal with benchmark packaged not available
def benchmark_factory() -> tuple[Optional[Type["BenchmarkSmoother"]],
                                 Optional[Type["BenchmarkVX"]],
                                 Optional[Type["BenchmarkVY"]]]:
    """
    Returns Benchmark Classes needed for benchmarking. Done via factory to avoid issues with the `benchmark` package
    not being available in a production environment.
    """
    pass