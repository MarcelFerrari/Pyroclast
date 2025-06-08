# TODO fill this in


### Benchmarking
Smoother Benchmarks rely on a config for the benchmarking. 

This config is searched by default in the `scripts/benchmark_config.json`. Override this property by setting the 
`PYROCLAST_BENCHMARK_CONFIG` environment variable to any valid config file. The environment variable takes precedence 
over the default path.

For more information about the content of the benchmark config, look at the `src/benchmark/config.py` file.