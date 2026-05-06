# PriorBAI
This repsository contains the code for the submisison titled: "Provable Sample Cost Reduction in Prior-Guided Hyperparameter Optimization". The paper introduces PriorBAI, a novel multi-fidelity Bayesian optimization algorithm that leverages prior knowledge to enhance hyperparameter optimization efficiency studies the effects of priors from a theoretical perspective, and conducts empirical evaluations.

## Installation
The package requires python 3.10 or higher. It is recommended to create a virtual environment before installing the package. You can install the package using pip:

```bash
git clone AnomyizedGitHubLink
pip install .
```

Note that to conduct experiments, you need to setup YahpoGym as described: [yahpogym's documentation](https://slds-lmu.github.io/yahpo_gym/getting_started.html#installation-python)


## Usage

Experiments are executed using the PyExperimenter library. To run experiments without a mysql database server, make sure that ```provider: mysql``` is set in all config files.

Experiments can then be executed using ``priorbai/priorbai.py``. To run Successive Halving make sure to use `conf/successive_halving.yml` as the experiment configuration file. To use PriorBAI with different priors, use `conf/experiment_config.yml`.

```python
if __name__ == "__main__":
    pyexp = PyExperimenter(
        experiment_configuration_file_path="conf/experiment_config.yml", # Select the correct config file here
        database_credential_file_path="conf/database_credentials.yml", # Not needed for sqlite
        use_codecarbon=False
    )
    pyexp.fill_table_from_config() # Creates the experiment table based on the config file
    pyexp.execute(run_experiment, max_experiments=30, random_order=True) # Execute the experiments
```



## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
