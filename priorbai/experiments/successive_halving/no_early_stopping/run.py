from __future__ import annotations

import logging
from pathlib import Path

from py_experimenter.experimenter import PyExperimenter

from priorbai.main import run_experiment

_HERE = Path(__file__).parent
_CREDENTIALS = _HERE.parents[3] / "conf" / "database_credentials.yml"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pyexp = PyExperimenter(
        experiment_configuration_file_path=str(_HERE / "config.yml"),
        database_credential_file_path=str(_CREDENTIALS),
        use_codecarbon=False,
    )
    # pyexp.reset_experiments("running", "error")
    # pyexp.fill_table_from_config()
    pyexp.execute(run_experiment, max_experiments=1, random_order=True)
