"""Sample 50000 configs per LCBench dataset and write the best val_accuracy to a JSON file."""
from __future__ import annotations

import json

import numpy as np
from yahpo_gym import benchmark_set, local_config

MAX_FIDELITY = 52
N_CONFIGS = 50000
SEED = 0
OUT_PATH = "data/global_optima.json"

local_config._config = {"data_path": "data/yahpogym/"}
benchmarkset = benchmark_set.BenchmarkSet("lcbench")

results: dict[int, float] = {}
for dataset_id, instance in enumerate(benchmarkset.instances):
    benchmarkset.set_instance(instance)
    cs = benchmarkset.get_opt_space(seed=SEED)
    cfg_list = cs.sample_configuration(size=N_CONFIGS)
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list]

    best = -np.inf
    for cfg in cfg_list:
        cfg_dict = cfg.get_dictionary()
        cfg_dict["epoch"] = MAX_FIDELITY
        acc = benchmarkset.objective_function(cfg_dict)[0]["val_accuracy"] / 100
        if acc > best:
            best = acc

    results[dataset_id] = float(best)
    print(f"dataset {dataset_id:2d} ({instance}): {best:.4f}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote global optima to {OUT_PATH}")
