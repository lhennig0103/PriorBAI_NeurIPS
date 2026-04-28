from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import ConfigSpace
from ConfigSpace.hyperparameters import (
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from priorbai.utils import Runhistory

logger = logging.getLogger(__name__)


class Benchmark(ABC):
    @abstractmethod
    def get_max_fidelity(self) -> int: ...

    @abstractmethod
    def sample(self, num_arms: int, seed: int, **kwargs: int) -> tuple[list[int], dict[int, float]]:
        """Sample num_arms configurations. Returns (arms, true_final_means)."""
        ...

    @abstractmethod
    def get_config(self, arm: int) -> Any:
        """Return the configuration dict (or identifier) associated with arm."""
        ...

    @abstractmethod
    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        """Evaluate the configuration associated with arm at the given fidelities."""
        ...


class SyntheticBenchmark(Benchmark):
    def __init__(self, max_fidelity: int, seed: int):
        self.max_fidelity = max_fidelity
        self.true_final_means: dict[int, float] = {}
        self.rng = np.random.default_rng(seed)

    def get_max_fidelity(self) -> int:
        return self.max_fidelity

    def sample(self, num_arms: int, seed: int, **kwargs) -> tuple[list[int], dict[int, float]]:
        final_means = sorted([self.rng.random() for _ in range(num_arms)], reverse=True)
        self.true_final_means = {arm: final_means[arm] for arm in range(num_arms)}
        return list(range(num_arms)), self.true_final_means

    def get_config(self, arm: int) -> int:
        return arm

    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        true_mu = self.true_final_means[arm]
        tau = 20.0 + 10.0 * arm
        return true_mu * (1.0 - np.exp(-fidelity_levels / tau))


class LCBenchBenchmark(Benchmark):
    MAX_FIDELITY = 52

    def __init__(self, dataset_id: int, seed: int, priorband: bool, n_prior_construction: int):
        from yahpo_gym import benchmark_set, local_config

        local_config._config = {"data_path": "data/yahpogym/"}
        self.benchmarkset = benchmark_set.BenchmarkSet("lcbench")
        self.benchmarkset.set_instance(self.benchmarkset.instances[dataset_id])
        self.configs: list[dict] = []
        self.configuration_space = self.benchmarkset.get_opt_space(seed=seed)
        self.priorband = priorband
        self.n_prior_construction = n_prior_construction
        if self.priorband:
            self.prior_configspace, self.prior_distribution = self.create_prior_configspace()

    def create_prior_configspace(self) -> tuple[ConfigSpace.ConfigurationSpace, dict[str, tuple[float, float]]]:
        # Step 1: sample n_prior_construction configurations
        cfg_list = self.configuration_space.sample_configuration(size=self.n_prior_construction)

        # Step 2: evaluate each at MAX_FIDELITY and select the best as prior center
        best_cfg = None
        best_acc = -np.inf
        for cfg in cfg_list:
            cfg_dict = cfg.get_dictionary()
            cfg_dict["epoch"] = self.MAX_FIDELITY
            acc = self.benchmarkset.objective_function(cfg_dict)[0]["val_accuracy"] / 100
            if acc > best_acc:
                best_acc = acc
                best_cfg = cfg.get_dictionary()

        # Step 3: build a new ConfigSpace with Normal distributions centred on best_cfg
        priro_configspace = ConfigSpace.ConfigurationSpace()
        prior_distribution: dict[str, tuple[float, float]] = {}
        for hp in self.configuration_space.get_hyperparameters():
            if hp.name == "epoch" or hp.name=="OpenML_task_id":
                priro_configspace.add_hyperparameter(hp)
                continue
            center = best_cfg[hp.name]
            if isinstance(hp, UniformFloatHyperparameter):
                sigma =  (hp.upper - hp.lower)/ 5
                priro_configspace.add_hyperparameter(ConfigSpace.NormalFloatHyperparameter(hp.name, mu=center, sigma=sigma, lower=hp.lower, upper=hp.upper))
            elif isinstance(hp, UniformIntegerHyperparameter):
                sigma = (hp.upper - hp.lower) / 5
                priro_configspace.add_hyperparameter(ConfigSpace.NormalIntegerHyperparameter(hp.name, mu=center, sigma=sigma, lower=hp.lower, upper=hp.upper))
            prior_distribution[hp.name] = (center, sigma)
        return priro_configspace, prior_distribution


    def get_max_fidelity(self) -> int:
        return self.MAX_FIDELITY

    def sample(self, num_arms: int, seed: int, **kwargs) -> tuple[list[int], dict[int, float]]:
        if num_arms == 0:
            self.configs = []
            return [], {}
        cfg_list = self.configuration_space.sample_configuration(size=num_arms)
        if not isinstance(cfg_list, list):
            cfg_list = [cfg_list]
        configs: list[tuple[dict, float]] = []
        for cfg in cfg_list:
            cfg_dict = cfg.get_dictionary()
            cfg_dict["epoch"] = self.MAX_FIDELITY
            configs.append(
                (cfg_dict, self.benchmarkset.objective_function(cfg_dict)[0]["val_accuracy"] / 100)
            )
        configs = sorted(configs, key=lambda c: c[1], reverse=True)
        self.configs = [cfg for cfg, _ in configs]
        arms = list(range(len(configs)))
        true_final_means: dict[int, float] = {arm: configs[arm][1] for arm in arms}
        return arms, true_final_means


    def prior_band_sampling(self, num_arms: int, seed: int, rung: int, eta: float, runhistory:Runhistory) -> tuple[list[int], dict[int, float]]:
        # TODO check whether we need to go through the brackets in which order? This is confusing
        rng = np.random.default_rng(seed)

        def activate_incumbent_sampling() -> bool:
            """
            We only support sequential execution.
            """
            return rung != 0
        
        def add_incumbent_sampling(chance_of_prior_configurations: float) -> tuple[float, float]:
            def how_likely_according_to_incumbent(config: dict, incumbent: dict) -> float:
                sigma = 0.25
                density = 1.0
                for hp in self.configuration_space.get_hyperparameters():
                    if hp.name in ("epoch", "OpenML_task_id"):
                        continue
                    hp_range = hp.upper - hp.lower
                    normalized_value = (config[hp.name] - hp.lower) / hp_range
                    normalized_incumbent = (incumbent[hp.name] - hp.lower) / hp_range
                    density *= np.exp(-0.5 * ((normalized_value - normalized_incumbent) / sigma) ** 2)
                return density
            
            def how_likely_according_to_prior(config: dict) -> float:
                density = 1.0
                for hp_name, (mu, sigma) in self.prior_distribution.items():
                    hp = self.configuration_space.get_hyperparameter(hp_name)
                    hp_range = hp.upper - hp.lower
                    normalized_value = (config[hp_name] - hp.lower) / hp_range
                    normalized_mu = (mu - hp.lower) / hp_range
                    normalized_sigma = sigma / hp_range
                    density *= np.exp(-0.5 * ((normalized_value - normalized_mu) / normalized_sigma) ** 2)
                return density

            priorband_relevant_configurations = runhistory.get_priorband_relevant_configurations()
            incumbent_configuration = priorband_relevant_configurations[0][0]

            incumbent_scores = 0
            prior_scores = 0
            for index, (config, _) in enumerate(priorband_relevant_configurations):
                weight = len(priorband_relevant_configurations) + 1 - index
                incumbent_scores += weight * how_likely_according_to_incumbent(config, incumbent_configuration)
                prior_scores += weight * how_likely_according_to_prior(config)

            chance_of_incumbent_configurations = chance_of_prior_configurations * (incumbent_scores / (incumbent_scores + prior_scores))
            chance_of_prior_configurations = chance_of_prior_configurations * (prior_scores / (incumbent_scores + prior_scores))
            return chance_of_prior_configurations, chance_of_incumbent_configurations

        def incumbent_sampling(incumbent: dict, n_configurations: int) -> tuple[list[dict], list[int], dict[int, float]]:
            configs = []
            true_final_means = {}
            for i in range(n_configurations):
                perturbed = dict(incumbent)
                for hp in self.configuration_space.get_hyperparameters():
                    if hp.name in ("epoch", "OpenML_task_id"):
                        continue
                    if rng.random() >= 0.5:
                        continue

                    # HP normalisation
                    hp_range = hp.upper - hp.lower
                    normalized = (incumbent[hp.name] - hp.lower) / hp_range

                    # Add noise
                    noise = rng.normal(0, 0.25)
                    adapted_hyperparameter = np.clip(normalized + noise, 0.0, 1.0)

                    # Denormalisation
                    value = adapted_hyperparameter * hp_range + hp.lower
                
                    if isinstance(hp, UniformIntegerHyperparameter):
                        value = int(round(value))
                    perturbed[hp.name] = value
                
                perturbed["epoch"] = self.MAX_FIDELITY
                acc = self.benchmarkset.objective_function(perturbed)[0]["val_accuracy"] / 100
                configs.append(dict(perturbed))
                true_final_means[i] = acc
            return configs, list(range(n_configurations)), true_final_means

        def prior_sampling(n_configurations: int) -> tuple[list[dict], list[int], dict[int, float]]:
            if n_configurations == 0:
                return [], [], {}
            configs = []
            true_final_means = {}
            samples = self.prior_configspace.sample_configuration(size=n_configurations)
            if not isinstance(samples, list):
                samples = [samples]
            for i, cfg in enumerate(samples):
                cfg_dict = cfg.get_dictionary()
                cfg_dict["epoch"] = self.MAX_FIDELITY
                acc = self.benchmarkset.objective_function(cfg_dict)[0]["val_accuracy"] / 100
                configs.append(cfg_dict)
                true_final_means[i] = acc
            return configs, list(range(n_configurations)), true_final_means
        
        chance_of_uniform_configurations = 1/ (1 + eta ** rung) 
        chance_of_prior_configurations = 1 - chance_of_uniform_configurations
        chance_of_incumbent_configurations = 0.0

        if activate_incumbent_sampling():
            chance_of_prior_configurations, chance_of_incumbent_configurations = add_incumbent_sampling(chance_of_prior_configurations)

        n_uniform, n_prior, n_incumbent = rng.multinomial(
            num_arms,
            [chance_of_uniform_configurations, chance_of_prior_configurations, chance_of_incumbent_configurations],
        )

        n_uniform = int(n_uniform)
        n_prior = int(n_prior)
        n_incumbent = int(n_incumbent)

        uniform_arms, uniform_means = self.sample(num_arms=n_uniform, seed=seed)
        uniform_configs = list(self.configs)

        prior_configs, prior_arms, prior_means = prior_sampling(n_prior)

        if n_incumbent > 0:
            incumbent = runhistory.get_priorband_relevant_configurations()[0][0]
            incumbent_configs, incumbent_arms, incumbent_means = incumbent_sampling(incumbent=incumbent, n_configurations=n_incumbent)

        else:
            incumbent_configs, incumbent_arms, incumbent_means = [], [], {}
            
        # Merge: re-index prior and incumbent arm IDs to avoid collisions
        offset_prior = len(uniform_arms)
        offset_incumbent = len(uniform_arms) + len(prior_arms)

        self.configs = uniform_configs + prior_configs + incumbent_configs

        all_means: dict[int, float] = {}
        all_means.update(uniform_means)
        all_means.update({arm + offset_prior: perf for arm, perf in prior_means.items()})
        all_means.update({arm + offset_incumbent: perf for arm, perf in incumbent_means.items()})

        all_arms = list(range(len(self.configs)))
        return all_arms, all_means
    

    def get_config(self, arm: int) -> dict:
        return self.configs[arm]

    def evaluate(self, arm: int, fidelity_levels: np.ndarray) -> np.ndarray:
        cfg = dict(self.configs[arm])
        res = []
        for fidelity in fidelity_levels:
            cfg["epoch"] = min(int(fidelity), self.MAX_FIDELITY)
            res.append(self.benchmarkset.objective_function(cfg)[0]["val_accuracy"] / 100)
        return np.array(res)


def get_benchmark(benchmark_name: str, num_arms: int, dataset_id: int, seed: int, priorband: bool, n_prior_construction: int) -> Benchmark:
    if benchmark_name == "synthetic":
        if priorband:
            raise ValueError("PriorBand is not supported for the synthetic benchmark.")
        return SyntheticBenchmark(max_fidelity=num_arms, seed=seed)
    elif benchmark_name == "lcbench":
        return LCBenchBenchmark(dataset_id=dataset_id, seed=seed, priorband=priorband, n_prior_construction=n_prior_construction)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name!r}")
