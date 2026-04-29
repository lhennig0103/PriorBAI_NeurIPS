from __future__ import annotations

import logging
import math
import warnings
from functools import partial
from collections.abc import Callable
from typing import Any
from priorbai.utils import Runhistory
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from priorbai.benchmarks import Benchmark, get_benchmark
from priorbai.kernels import get_kernel
from priorbai.priors import get_prior_means

logger = logging.getLogger(__name__)


def prior_guided_successive_halving(
        benchmark: Benchmark,
        num_arms: int,
        prior_kind: str,
        sampling_seed: int,
        budget_N: int,
        eta: int,
        kernel: Kernel | None,
        use_predicted_y: bool,
        use_early_stopping: bool,
        delta: float,
        epsilon: float,
        sigma0_sq: float,
        rng: np.random.Generator,
        seed: int,
        result_processor: Any | None,
        hb_bracket: int | None,
        runhistory: Runhistory,
        sample_configurations: Callable[..., tuple[list[int], dict[int, float]]],
) -> tuple[float, int, int, dict[int, float]]:

    arms, true_final_means = sample_configurations(num_arms, sampling_seed)
    T_max = benchmark.get_max_fidelity()
    prior_means = get_prior_means(arms, prior_kind, true_final_means, epsilon, rng)

    number_of_arms = len(arms)
    if number_of_arms < 1:
        raise ValueError("At least one arm is required.")

    active_arms = arms.copy()
    number_of_rounds = math.ceil(math.log(number_of_arms, eta))
    mu_hat: dict[int, float] = {arm: prior_means[arm] for arm in arms}
    budget_consumed = 0
    previous_round_budget = 0
    arm_ts: dict[int, list[int]] = {arm: [] for arm in arms}
    arm_ys: dict[int, list[float]] = {arm: [] for arm in arms}
    N_stop = 0.0
    round_index = -1
    stopped_early = False
    C_log = math.log(2.0 * math.log(number_of_arms, eta) * ((number_of_arms / 2) - 1) / delta)
    logger.debug("C_log=%s", C_log)

    for round_index in range(number_of_rounds):
        if len(active_arms) == 0:
            logger.warning("S_r is empty at round %d — this should not happen.", round_index)
            break

        round_budget = math.floor(budget_N / (number_of_rounds * len(active_arms)))
        if round_budget <= previous_round_budget:
            logger.debug("No additional budget to allocate at round %d; stopping.", round_index)
            break

        budget_consumed += len(active_arms) * (round_budget - previous_round_budget)

        # 1. Observation & Extrapolation
        round_y: dict[int, float] = {}
        round_mu_hat: dict[int, float] = {}
        round_sigma_sq: dict[int, float] = {}
        Sigma = 0.0

        for arm in active_arms:
            new_fidelity_levels = np.arange(previous_round_budget + 1, round_budget + 1, dtype=int)
            if len(new_fidelity_levels) > 0:
                new_ys = benchmark.evaluate(arm, new_fidelity_levels)
                if len(new_ys) != len(new_fidelity_levels):
                    raise ValueError("evaluate must return one y per t.")
                arm_ts[arm].extend(new_fidelity_levels.tolist())
                arm_ys[arm].extend(new_ys.tolist())

            X = np.asarray(arm_ts[arm], dtype=float).reshape(-1, 1) / T_max
            y = np.asarray(arm_ys[arm], dtype=float)
            logger.debug("arm=%s  X=%s  y=%s", arm, X, y)

            nu_j = prior_means[arm]

            if len(X) == 0:
                mu_j_r = nu_j
                sigma_j_r_sq = sigma0_sq
            else:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.05 ** 2,
                    normalize_y=False,
                    random_state=seed,
                )
                gp.fit(X, y)
                X_star = np.array([[1.0]])
                mu_res, std_res = gp.predict(X_star, return_std=True)
                mu_j_r = min(1.0, float(mu_res.item()))
                sigma_j_r_sq = float(std_res.item() ** 2)
                Sigma += sigma_j_r_sq

                logger.debug(
                    "arm=%s  n_r=%d  mu_j_r=%.4f  sigma_j_r_sq=%.4f  actual=%.4f",
                    arm, round_budget, mu_j_r, sigma_j_r_sq, arm_ys[arm][-1],
                )

            round_y[arm] = arm_ys[arm][-1]
            round_mu_hat[arm] = mu_j_r
            round_sigma_sq[arm] = sigma_j_r_sq

        logger.debug("round_sigma_sq=%s  Sigma=%.6f", round_sigma_sq, Sigma)

        for arm in active_arms:
            runhistory.add(round_budget, benchmark.get_config(arm), round_y[arm])

        previous_round_budget = round_budget
        mu_hat.update(round_mu_hat)

        # 2. Candidate Selection
        i_hat = max(active_arms, key=lambda a: round_mu_hat[a])

        # 3. Stopping Condition
        Deltas: dict[int, float] = {}
        for arm in active_arms:
            if arm == i_hat:
                continue
            performance_gap = round_mu_hat[i_hat] - round_mu_hat[arm]
            Deltas[arm] = max(epsilon, performance_gap)

        logger.debug("round_mu_hat=%s", round_mu_hat)

        if Deltas:
            N_stop_candidates = []
            nu_i = prior_means[i_hat]
            for arm, Delta_j_r in Deltas.items():
                nu_j = prior_means[arm]
                logger.debug(
                    "arm=%s  nu_i=%.4f  nu_j=%.4f  nu_diff=%.4f",
                    arm, nu_i, nu_j, nu_i - nu_j,
                )
                formula_bracket = C_log - ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                formula_bracket = max(formula_bracket, 0.0)
                if Delta_j_r <= 0:
                    continue

                N_stop_j = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * formula_bracket
                base_budget = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * C_log
                budget_reduction = (4.0 * number_of_rounds * Sigma / (Delta_j_r ** 2)) * ((nu_i - nu_j) * Delta_j_r) / (2.0 * sigma0_sq)
                logger.debug(
                    "arm=%s  r=%d  number_of_rounds=%d  Sigma=%.6f  Delta=%.6f  bracket=%.6f  N_stop=%.4f  "
                    "base_budget=%.4f  budget_reduction=%.4f",
                    arm, round_index, number_of_rounds, Sigma, Delta_j_r, hb_bracket, N_stop_j, base_budget, budget_reduction,
                )
                N_stop_candidates.append(N_stop_j)

            if N_stop_candidates:
                N_stop_arr = np.array(N_stop_candidates)
                logger.debug(
                    "N_stop  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                    N_stop_arr.min(), N_stop_arr.max(), N_stop_arr.mean(), N_stop_arr.std(),
                )

            N_stop = max(N_stop_candidates) if N_stop_candidates else 0.0
            logger.debug("N_used=%d  N_stop=%.4f", budget_consumed, N_stop)

            if result_processor is not None:
                result_processor.process_logs({
                    "sh_iterations": {
                        "bracket": hb_bracket,
                        "iteration": round_index,
                        "fidelity": round_budget,
                        "num_arms": len(active_arms),
                        "best_arm_included": 1 if 0 in active_arms else 0,
                        "budget_spent_so_far": budget_consumed,
                        "N_stop": N_stop,
                    }
                })

            if use_early_stopping and budget_consumed >= N_stop:
                logger.debug(
                    "Stopping condition reached at round %d/%d with %d arms remaining.",
                    round_index, number_of_rounds, len(active_arms),
                )
                best = i_hat
                break

        # 4. Pruning
        if use_predicted_y:
            S_r_sorted = sorted(active_arms, key=lambda a: round_mu_hat[a], reverse=True)
        else:
            S_r_sorted = sorted(active_arms, key=lambda a: round_y[a], reverse=True)

        number_of_arms_to_keep = math.ceil(len(S_r_sorted) / eta)
        active_arms = S_r_sorted[:number_of_arms_to_keep]

    else:
        best = max(arms if len(active_arms) == 0 else active_arms, key=lambda a: mu_hat[a])

    if result_processor is not None:
        result_processor.process_logs({
            "brackets": {
                "bracket": hb_bracket if hb_bracket is not None else 0,
                "n_arms": number_of_arms,
                "budget_used": budget_consumed,
                "stopped_early": 1 if stopped_early else 0,
                "stopped_after_round": round_index,
            }
        })
    return true_final_means[best], budget_consumed, len(active_arms), true_final_means


def prior_guided_hyperband(
        benchmark: Benchmark,
        prior_kind: str,
        eta: int,
        runhistory: Runhistory,
        kernel: Kernel | None,
        use_predicted_y: bool,
        use_early_stopping: bool,
        delta: float,
        epsilon: float,
        sigma0_sq: float,
        rng: np.random.Generator,
        seed: int,
        result_processor: Any | None,
        sample_configurations: Callable[..., tuple[list[int], dict[int, float]]],
) -> tuple[float, int, int, dict[int, dict[int, float]]]:
    T_max = benchmark.get_max_fidelity()
    s_max = math.floor(math.log(T_max, eta)) if T_max > 1 else 0
    B = (s_max + 1) * T_max

    total_budget = 0
    true_final_means: dict[int, dict[int, float]] = {}
    bracket_winner_perfs: list[float] = []
    seed_sequence = np.random.SeedSequence(seed)
    for rung, s in enumerate(range(s_max, -1, -1)):
        num_arms = max(1, math.ceil(B / T_max * eta ** s / (s + 1)))

        logger.debug(
            "Hyperband bracket s=%d/%d: n_s=%d arms, r_s=%.1f, B=%d",
            s, s_max, num_arms, T_max / eta ** s, B,
        )

        sample_configurations_initialised = partial(sample_configurations, rung=rung, eta=eta, runhistory=runhistory)

        winner_perf, budget_used, _, bracket_true_final_means = prior_guided_successive_halving(
            benchmark=benchmark,
            num_arms=num_arms,
            prior_kind=prior_kind,
            sampling_seed=seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint32)[0],
            budget_N=B,
            eta=eta,
            hb_bracket=s,
            kernel=kernel,
            use_predicted_y=use_predicted_y,
            use_early_stopping=use_early_stopping,
            delta=delta,
            epsilon=epsilon,
            sigma0_sq=sigma0_sq,
            rng=rng,
            seed=seed,
            runhistory=runhistory,
            result_processor=result_processor,
            sample_configurations=sample_configurations_initialised,
        )
        total_budget += budget_used
        true_final_means[s] = bracket_true_final_means

        bracket_winner_perfs.append(winner_perf)
        logger.debug("Bracket s=%d  winner_perf=%.4f", s, winner_perf)

    best_winner_perf = max(bracket_winner_perfs)
    return best_winner_perf, total_budget, 1, true_final_means


def setup_run(config):
    benchmark_name = config["benchmark"]
    num_arms = int(config["num_arms"])
    dataset_id = int(config["dataset_id"])
    kernel_name = config.get("kernel", "satexp_rbf")
    seed = int(config["seed"])
    use_priorband = bool(config.get("optimizer") == "priorband")
    
    benchmark = get_benchmark(benchmark_name, num_arms, dataset_id, seed, priorband=use_priorband, n_prior_construction=1000, n_reference_configs=5000)
    learning_curve_kernel = get_kernel(kernel_name)

    return benchmark, learning_curve_kernel


def run_experiment(config, result_processor, custom_config):
    logger.debug("config: %s", config)

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    sigma0 = float(config["sigma0"])
    sigma0_sq = sigma0 ** 2
    epsilon = float(config["epsilon"])
    delta = float(config["delta"])
    use_predicted_y = bool(config["use_predicted_y"])
    use_early_stopping = bool(config.get("use_early_stopping", False))

    num_arms = int(config["num_arms"])
    prior_kind = config["prior_kind"]
    optimizer = config["optimizer"]

    benchmark, learning_curve_kernel = setup_run(config)
    T_max = benchmark.get_max_fidelity()

    eta = int(config.get("eta", 2))
    runhistory = Runhistory(eta=eta)
    shared_kwargs = dict(
        benchmark=benchmark,
        prior_kind=prior_kind,
        kernel=learning_curve_kernel,
        use_predicted_y=use_predicted_y,
        use_early_stopping=use_early_stopping,
        delta=delta,
        epsilon=epsilon,
        sigma0_sq=sigma0_sq,
        rng=rng,
        seed=seed,
        runhistory=runhistory,
        result_processor=result_processor,
    )

    if optimizer == "successive_halving":
        winner_perf, budget_used, num_arms_left, bracket_perfs = prior_guided_successive_halving(
            num_arms=num_arms,
            sampling_seed=seed,
            budget_N=num_arms * np.log2(num_arms),
            eta=eta,
            hb_bracket=None,
            sample_configurations=benchmark.sample,
            **shared_kwargs,
        )
        true_final_means = {0: bracket_perfs}
    elif optimizer == "hyperband":
        winner_perf, budget_used, num_arms_left, true_final_means = prior_guided_hyperband(
            eta=eta,
            sample_configurations=benchmark.sample,
            **shared_kwargs,
        )
    elif optimizer == "priorband":
        winner_perf, budget_used, num_arms_left, true_final_means = prior_guided_hyperband(
            eta=eta,
            sample_configurations=benchmark.prior_band_sampling,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Choose 'successive_halving' or 'hyperband'.")

    all_perfs = [p for bracket in true_final_means.values() for p in bracket.values()]
    local_best = max(all_perfs)
    num_epsilon_optimal_arms = sum(1 for p in all_perfs if local_best - p <= epsilon)
    regret = benchmark.global_optimum - winner_perf

    result_processor.process_results({
        "T_max": T_max,
        "consumed_budget": budget_used,
        "remaining_arms": num_arms_left,
        "num_epsilon_optimal_arms": num_epsilon_optimal_arms,
        "regret": regret,
        "epsilon_optimal": 1 if regret <= epsilon else 0,
    })
