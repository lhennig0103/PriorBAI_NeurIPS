import math
import tqdm
from typing import Callable, List, Dict, Any, Sequence

from py_experimenter.experimenter import PyExperimenter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, RBF
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from yahpo_gym import benchmark_set, local_config

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class SaturatingExpKernel(Kernel):
    """
    Saturating exponential basis kernel:

        s(t) = 1 - exp(-t / tau)
        k(t, t') = sigma_sq * s(t) * s(t')

    This encodes functions that are linear combinations of a saturating,
    increasing shape. You typically combine this with another kernel
    (e.g., RBF) for extra flexibility.
    """

    def __init__(
        self,
        tau: float = 0.3,
        tau_bounds=(0.01, 1.0),
        sigma_sq: float = 1.0,
        sigma_sq_bounds=(1e-3, 1e3),
    ):
        self.tau = float(tau)
        self.tau_bounds = tau_bounds
        self.sigma_sq = float(sigma_sq)
        self.sigma_sq_bounds = sigma_sq_bounds

    @property
    def hyperparameter_tau(self):
        # fixed by default; you can make this learnable by setting bounds.
        return Hyperparameter("tau", "numeric", self.tau_bounds)

    @property
    def hyperparameter_sigma_sq(self):
        return Hyperparameter("sigma_sq", "numeric", self.sigma_sq_bounds)

    def _s(self, X):
        X = np.atleast_2d(X)
        t = X[:, 0]
        s = 1.0 - np.exp(-t / self.tau)
        return s.reshape(-1, 1)

    def __call__(self, X, Y=None, eval_gradient=False):
        sX = self._s(X)
        if Y is None:
            sY = sX
        else:
            sY = self._s(Y)

        K = self.sigma_sq * (sX @ sY.T)  # outer product

        if not eval_gradient:
            return K

        # Gradients only for X == Y, as required by sklearn
        if Y is not None and Y is not X:
            raise ValueError("eval_gradient=True only supported for Y is None (Y == X).")

        # Gradient wrt log sigma_sq: ∂K/∂theta_sigma = K
        dK_dtheta_sigma = K

        # tau is fixed by default → no gradient
        K_grad = np.stack([dK_dtheta_sigma], axis=2)  # (n, n, 1)

        return K, K_grad

    def diag(self, X):
        sX = self._s(X)
        return self.sigma_sq * (sX[:, 0] ** 2)

    def is_stationary(self):
        return False  # non-stationary in t

    def __repr__(self):
        return f"SaturatingExpKernel(tau={self.tau}, sigma_sq={self.sigma_sq})"


def mf_prior_guided_successive_halving(
        arms: Sequence[Any],
        budget_N: int,
        prior_means: Dict[Any, float],
        T_max: int,
        observe_fn: Callable[[Any, np.ndarray], np.ndarray],
        kernel: Kernel | None = None,
        delta: float = 0.1,
        epsilon: float = 0.01,
        sigma0_sq: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
        result_processor: Any | None = None
) -> tuple[Any, int, int]:

    arms = list(arms)
    K = len(arms)
    if K < 1:
        raise ValueError("At least one arm is required.")
    if verbose:
        print("Initialize")
    S_r = arms.copy()
    R = math.ceil(math.log2(K))
    if verbose:
        print("Per-arm latest posterior mean estimate")
    mu_hat: Dict[Any, float] = {arm: prior_means[arm] for arm in arms}

    if verbose:
        print("Track total used budget and previous fidelity level")
    N_used = 0
    n_prev = 0

    if verbose:
        print("Store observed learning curves per arm: t -> y")
        print("We'll maintain lists of fidelities and corresponding observations.")
    arm_ts: Dict[Any, List[int]] = {arm: [] for arm in arms}
    arm_ys: Dict[Any, List[float]] = {arm: [] for arm in arms}

    rng = np.random.RandomState(random_state)

    if verbose:
        print("Precompute constant in stopping condition")
    C_log = math.log(2.0 * math.log2(K)*(K/2 - 1) / delta)

    if verbose:
        print("C_log", C_log)

    for r in range(R):
        if len(S_r) == 0:
            print("Should not happen, but for safety")
            break

        # Per-round per-arm fidelity
        n_r = math.floor(budget_N / (R * len(S_r)))
        if n_r <= n_prev:
            print("No additional budget to allocate; break and return best seen so far")
            break
        # print("Budget for round ", r, ": ", n_r)
        # Update total used budget
        N_used += len(S_r) * (n_r - n_prev)

        # 1. Observation & Extrapolation
        round_mu_hat: Dict[Any, float] = {}
        round_sigma_sq: Dict[Any, float] = {}
        # print("Number of arms", len(S_r))

        Sigma = 0
        for arm in S_r:
            # Observe learning curve up to fidelity n_r:
            # Only query new fidelities t in (n_prev, n_r].
            new_ts = np.arange(n_prev + 1, n_r + 1, dtype=int)
            if len(new_ts) > 0:
                new_ys = observe_fn(arm, new_ts)
                if len(new_ys) != len(new_ts):
                    raise ValueError("observe_fn must return one y per t.")

                arm_ts[arm].extend(new_ts.tolist())
                arm_ys[arm].extend(new_ys.tolist())

            # Fit GP on residuals y - nu_j with zero-mean
            X = np.asarray(arm_ts[arm], dtype=float).reshape(-1, 1)/T_max
            y = np.asarray(arm_ys[arm], dtype=float)
            if verbose:
                print("X", X)
                print (y)

            nu_j = prior_means[arm]

            if len(X) == 0:
                # No data; stick with prior
                mu_j_r = nu_j
                sigma_j_r_sq = sigma0_sq
            else:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.05**2,
                    normalize_y=False,
                    random_state=rng,
                )
                gp.fit(X, y)


                X_star = np.array([[float(1)]])
                mu_res, std_res = gp.predict(X_star, return_std=True)
                mu_j_r = min(1.0, mu_res.item())
                sigma_j_r_sq = float(std_res.item() ** 2)
                Sigma += sigma_j_r_sq

                if verbose and arm in [0,1]:
                    print(y.tolist())
                    print(gp.predict(X).tolist())
                    print(arm, n_r, "\t\t", mu_j_r, sigma_j_r_sq, "Actual value", arm_ys[arm][-1])

            round_mu_hat[arm] = mu_j_r
            round_sigma_sq[arm] = sigma_j_r_sq

        if verbose:
            print("sigmas", round_sigma_sq)
            print("Sigma", Sigma)

        # Update previous fidelity
        n_prev = n_r
        mu_hat.update(round_mu_hat)

        # 2. Candidate Selection
        # Best arm by predicted mean in this round
        i_hat = max(S_r, key=lambda a: round_mu_hat[a])

        # 3. Stopping Condition
        Deltas: Dict[Any, float] = {}
        for arm in S_r:
            if arm == i_hat:
                continue
            gap = round_mu_hat[i_hat] - round_mu_hat[arm]
            Deltas[arm] = max(epsilon, gap)

        if verbose:
            print("round_mu_hat", round_mu_hat)

        if Deltas:
            N_stop_candidates = []
            nu_i = prior_means[i_hat]
            for arm in Deltas:
                Delta_j_r = Deltas[arm]
                nu_j = prior_means[arm]
                if verbose:
                    print("nu_i", nu_i, "nu_j", nu_j, "nu diff", nu_i - nu_j)
                bracket = C_log - ((nu_i - nu_j) * Delta_j_r ) / (2.0 * sigma0_sq)
                # If the bracket is negative, clamp to 0 (no extra budget needed in theory)
                bracket = max(bracket, 0.0)
                if Delta_j_r <= 0:
                    continue

                N_stop_j = (4.0 * R * Sigma / (Delta_j_r ** 2)) * bracket

                # show effect sizes
                base_budget = (4.0 * R * Sigma / (Delta_j_r ** 2))  * C_log
                budget_reduction = (4.0 * R * Sigma / (Delta_j_r ** 2)) * ((nu_i - nu_j) * Delta_j_r ) / (2.0 * sigma0_sq)
                if verbose:
                    print("Variance adaptive base budget", base_budget, " Budget reduction due to prior ", budget_reduction)

                if verbose:
                    print("arm", arm, "\t\t",  "r", r, "R", R, "Sigma", round(Sigma,6), "Delta_j_r", round(Delta_j_r, 6), "\t\t =>", "bracket", bracket, "value", round((4 * R * Sigma)/ Delta_j_r**2,6), "\t\t\t =>", "N_stop", N_stop_j)
                N_stop_candidates.append(N_stop_j)

            N_stop_arr = np.array(N_stop_candidates)
            if verbose:
                print("N_stop statistics", "Min", N_stop_arr.min(), "Max", N_stop_arr.max(), "Mean", N_stop_arr.mean(), "Std", N_stop_arr.std())

            # print("N_stop_candidates: ", N_stop_candidates)
            N_stop = max(N_stop_candidates) if N_stop_candidates else 0.0
            if verbose:
                print("N_used", N_used, "but N_stop requires", N_stop)

            if result_processor is not None:
                result_processor.process_logs({
                    "sh_iterations": {
                        "iteration": r,
                        "num_arms": len(S_r),
                        "best_arm_included": 1 if 0 in S_r else 0,
                        "budget_spent_so_far": N_used,
                        "N_stop": N_stop
                    }
                })

            if N_used >= N_stop:
                if verbose:
                    print("Stopping condition reached; return i_hat; current round was ", r, " out of ", R)
                return i_hat, N_used, len(S_r)

        # 4. Pruning
        # Keep top ceil(|S_r| / 2) arms by predicted means
        S_r_sorted = sorted(S_r, key=lambda a: round_mu_hat[a], reverse=True)
        # print("S_r_sorted", S_r_sorted)
        keep = math.ceil(len(S_r_sorted) / 2.0)
        # print(keep)
        S_r = S_r_sorted[:keep]

        if 0 not in S_r and verbose:
            print("KICKED OUT THE BEST ARM!")

    # If we did not stop early, return the best arm among survivors
    if len(S_r) == 0:
        # Fall back to best overall by last mu_hat
        return max(arms, key=lambda a: mu_hat[a]), N_used, len(S_r)
    else:
        return max(S_r, key=lambda a: mu_hat[a]), N_used, len(S_r)

def run_experiment(config, result_processor, custom_config):
    seed = int(config['seed'])
    # set seed
    np.random.seed(seed)

    benchmark = config['benchmark']
    num_arms = int(config['num_arms'])

    prior = config['prior']
    sigma0 = float(config['sigma0'])
    sigma0_sq = sigma0 ** 2

    epsilon = float(config['epsilon'])
    delta = float(config['delta'])


    ### TODO: Does this need to be configurable?
    prior_std = 0.01

    true_final_means: Dict[Any, float] = {}

    if benchmark == 'synthetic':
        T_max = num_arms
        final_means = sorted([np.random.rand() for arm in range(num_arms)], reverse=True)
        true_final_means = {arm: final_means[arm] for arm in range(num_arms)}
        def synthetic_learning_curve(arm: int, t: np.ndarray) -> np.ndarray:
            """
            Toy environment: each arm has a logistic-ish learning curve
            approaching its true final mean.
            """
            true_mu = true_final_means[arm]
            # Simple saturating curve: f(t) = true_mu * (1 - exp(-t / tau))
            tau = 20.0 + 10.0 * arm  # just to make arms differ a bit
            mean_vals = true_mu * (1.0 - np.exp(-t / tau))
            return mean_vals
        eval_fun = synthetic_learning_curve
    elif benchmark == 'lcbench':
        yahpogym_folder = "data/yahpogym/"
        local_config.set_data_path(yahpogym_folder)
        T_max = 52
        benchmark = benchmark_set.BenchmarkSet("lcbench")
        benchmark.set_instance(benchmark.instances[seed % len(benchmark.instances)])
        configs = []

        config_list = benchmark.get_opt_space().sample_configuration(size=num_arms)
        for config in config_list:
            config_dict = config.get_dictionary()
            config_dict["epoch"] = T_max
            configs.append((config_dict, benchmark.objective_function(config_dict)[0]["val_accuracy"]/100))
        configs = sorted(configs, key=lambda c: c[1], reverse=True)
        true_final_means = {arm: configs[arm][1] for arm in range(num_arms)}

        class YahpoGymeEvaluator:
            def __init__(self, benchmark, configs):
                self.benchmark = benchmark
                self.configs = configs

            def evaluate(self, arm: int, t: np.ndarray) -> np.ndarray:
                cfg = self.configs[arm][0]
                res = []
                for ti in t:
                    print(ti)
                    cfg["epoch"] = int(ti)
                    res += [self.benchmark.objective_function(cfg)[0]["val_accuracy"]/100]
                return np.array(res)
        eval = YahpoGymeEvaluator(benchmark, configs)
        eval_fun = eval.evaluate

    arms = list(true_final_means.keys())

    prior_means = {}
    max_true_mean = np.array(list(true_final_means.values())).max()
    if prior == "uniform":
        for arm in arms:
            prior_means[arm] = np.array(list(true_final_means.values())).mean()
    elif prior == "rank":
        for arm in arms:
            # prior is the inverse of the true mean's rank
            prior_means[arm] = 1 / (arm+1)
    elif prior == "performance":
        # prior mean sampled from a normal distribution around the actual mean
        for arm in arms:
            prior_means[arm] = min(1.0, np.random.normal(true_final_means[arm], prior_std))
    elif prior == "inverse_rank":
        # prior is the inverse of the true mean's rank
        for arm in arms:
            prior_means[arm] = (arm+1) / num_arms
    elif prior == "indicator":
        # prior is the inverse of the true mean's rank
        for arm in arms:
            prior_means[arm] = 1 if (max_true_mean - true_final_means[arm]) <= epsilon else 0
    else:
        raise ValueError("Unknown prior type")

    # create an IPL kernel
    # ipl_kernel = InversePowerLawKernel(length_scale=2)
    sat_kernel = SaturatingExpKernel(tau=0.3, sigma_sq=1.0)
    smooth_kernel = RBF(length_scale=0.2)
    lc_kernel = sat_kernel + smooth_kernel

    selected_best, budget_used, num_arms_left = mf_prior_guided_successive_halving(
        arms=arms,
        budget_N=num_arms*np.log2(num_arms),
        prior_means=prior_means,
        T_max=T_max,
        observe_fn=eval_fun,
        kernel=lc_kernel,      # <-- use IPL here
        delta=delta,
        epsilon=epsilon,
        sigma0_sq=sigma0_sq,
        verbose=False,
        result_processor=result_processor
    )
    actual_best = max(true_final_means, key=true_final_means.get)

    # count the epsilon optimal arms in the set
    num_epsilon_optimal_arms = 0
    for arm in arms:
        if max_true_mean - true_final_means[arm] <= epsilon:
            num_epsilon_optimal_arms += 1

    # compute regret of selected arm
    regret = max_true_mean-true_final_means[selected_best]

    result_processor.process_results(
        {
            "T_max": T_max,
            "consumed_budget": budget_used,
            "remaining_arms": num_arms_left,
            "num_epsilon_optimal_arms": num_epsilon_optimal_arms,
            "arm_id_selected": selected_best,
            "regret": regret,
            "epsilon_optimal": 1 if regret <= epsilon else 0,
            "best_arm": 1 if actual_best == selected_best else 0
        }
    )

if __name__ == "__main__":
    pyexp = PyExperimenter(
        experiment_configuration_file_path="conf/experiment_config.yml",
        database_credential_file_path="conf/database_credentials.yml",
        use_codecarbon=False
    )

    #pyexp.fill_table_from_config()
    pyexp.execute(run_experiment, max_experiments=40, random_order=True)

    # class MockupProcesor:
    #     def process_results(self, data):
    #         print(data)
    #
    # run_experiment({
    #     "seed": 0,
    #     "benchmark": "lcbench",
    #     "prior": "uniform",
    #     "sigma0": 0.1,
    #     "epsilon": 0.01,
    #     "delta": 0.05,
    #     "num_arms": 32,
    # }, MockupProcesor(), {})
