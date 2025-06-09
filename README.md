# TODO complete. Is wip

### Setup
This project uses [Poetry](https://python-poetry.org/) as its dependency manager as it offers more robust and 
reproducible dependency resolution even across platforms and ISAs.

Assuming you're coming from `pip` / `venv`, and you are used to your virtual environments being stored inside your 
project directory, we suggest changing the poetry config.

This command will set the install directory of the venv inside the project. Poetry will create a `.venv` folder in the
project root dir for you.
```bash
poetry config virtualenvs.in-project true
```

After you've configured Poetry to your liking, it's time to get the dependencies. There are two important concepts to 
be aware of when installing:
- Extras
- Groups

**Extras**: keywords that can be added to the install command of a python package that installs additional dependencies. 
E.g. `fastapi[multipart]` installs the `fastapi` with all all necessary further dependencies to parse `multipart` forms.

**Groups**: Are a feature provided by Poetry. They work the same as extras but are unique to Poetry. I.e. if you run 
`poetry install --with test` and you've defined a group called `test`, poetry will install your project dependencies as 
well as the dependencies specified in the `test` group. However, this group isn't mapped into pip. 
You cannot run `pip install package[test]`.

To be able to run Pyroclast with the benchmarks, install all groups of the project with `poetry install --all-groups`

Now everything should be setup and running.

There are two scripts that you need in order to run benchmarks of the smoother:
- `src/benchmark/runner.py` (performs the task of running the benchmark)
- `src/benchmark/printer.py` (Loads results and prints to shell.)

### Benchmarking
Smoother Benchmarks rely on a config for the benchmarking. 

This config is searched by default in the `scripts/benchmark_config.json`. Override this property by setting the 
`PYROCLAST_BENCHMARK_CONFIG` environment variable to any valid config file. **The environment variable takes precedence 
over the default path.**

For more information about the content of the benchmark config, look at the `src/benchmark/config.py` file.

