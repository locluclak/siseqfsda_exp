
import importlib.util
import os
import sys
from pathlib import Path


def is_colab() -> bool:
    return (
        "COLAB_RELEASE_TAG" in os.environ
        or "COLAB_GPU" in os.environ
        or "google.colab" in sys.modules
        or Path("/content").exists()
    )


def load_custom_linprog(custom_dir: Path) -> None:
    import scipy.optimize

    simplex_path = custom_dir / "_linprog_simplex.py"
    spec_simplex = importlib.util.spec_from_file_location(
        "scipy.optimize._linprog_simplex", simplex_path
    )
    mod_simplex = importlib.util.module_from_spec(spec_simplex)
    sys.modules["scipy.optimize._linprog_simplex"] = mod_simplex
    spec_simplex.loader.exec_module(mod_simplex)

    linprog_path = custom_dir / "_linprog.py"
    spec_linprog = importlib.util.spec_from_file_location(
        "scipy.optimize._linprog", linprog_path
    )
    mod_linprog = importlib.util.module_from_spec(spec_linprog)
    sys.modules["scipy.optimize._linprog"] = mod_linprog
    spec_linprog.loader.exec_module(mod_linprog)

    setattr(scipy.optimize, "_linprog_simplex", mod_simplex)
    setattr(scipy.optimize, "_linprog", mod_linprog)
    if hasattr(mod_linprog, "linprog"):
        setattr(scipy.optimize, "linprog", mod_linprog.linprog)


if is_colab():
    custom_dir = Path(__file__).resolve().parent / "files_to_replace"
    if custom_dir.exists():
        load_custom_linprog(custom_dir)

import os
import argparse
import numpy as np
from scipy.stats import skewnorm

from modules import DA_SeqFS, SI_SeqFS_DA


ALPHA = 0.05
REPETITIONS = 200

NS_LIST = [50, 100, 150, 200]
NT = 20
P = 5
K = 3

TRUE_BETA_S = np.full((P, 1), 2.0)
TRUE_BETA_T = np.zeros((P, 1))

DEFAULT_OUT_DIR = os.path.join("new_exp", "nonnormal")

SKEW_PARAM = 10.0
SKEW_MEAN, SKEW_VAR = skewnorm.stats(SKEW_PARAM, moments="mv")
LAPLACE_SCALE = 1.0 / np.sqrt(2.0)
T20_STD = np.sqrt(20.0 / 18.0)

SETUPS = [
    ("laplace", "laplace", "known"),
    ("skewnorm10", "skewnorm10", "known"),
    ("t20", "t20", "known"),
    ("normal_estvar", "normal", "estimated"),
]

SETUP_NAMES = [setup[0] for setup in SETUPS]

METHODS = [
    ("forward_fixedK3", K),
    # ("forward_aic", "AIC"),
]


def sample_noise(dist, n):
    if dist == "normal":
        return np.random.normal(0.0, 1.0, size=(n, 1))
    if dist == "laplace":
        return np.random.laplace(0.0, LAPLACE_SCALE, size=(n, 1))
    if dist == "t20":
        return np.random.standard_t(20, size=(n, 1)) / T20_STD
    if dist == "skewnorm10":
        raw = skewnorm.rvs(SKEW_PARAM, size=(n, 1))
        return (raw - SKEW_MEAN) / np.sqrt(SKEW_VAR)
    raise ValueError(dist)


def generate(ns, nt, p, true_beta_s, true_beta_t, dist):
    Xs = np.random.normal(0.0, 1.0, size=(ns, p))
    Xt = np.random.normal(0.0, 1.0, size=(nt, p))
    Ys = Xs.dot(true_beta_s) + sample_noise(dist, ns)
    Yt = Xt.dot(true_beta_t) + sample_noise(dist, nt)
    return Xs, Xt, Ys, Yt


def residual_variance(Y, X):
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X.dot(coef)
    return float(resid.T.dot(resid).item()) / (X.shape[0] - X.shape[1])
def mean_variance(Y):
    y = np.asarray(Y, dtype=float).reshape(-1)
    n = y.shape[0]
    if n <= 1:
        raise ValueError("Cần ít nhất 2 mẫu để tính phương sai mẫu không chệch.")
    diff = y - np.mean(y)
    return float(np.dot(diff, diff)) / (n - 1)

def build_sigma(Xs, Xt, Ys, Yt, variance_mode):
    ns, nt = Xs.shape[0], Xt.shape[0]
    if variance_mode == "known":
        return np.identity(ns), np.identity(nt)
    # v_s = residual_variance(Ys, Xs)
    # v_t = residual_variance(Yt, Xt)
    v_s = mean_variance(Ys)
    v_t = mean_variance(Yt)
    return v_s * np.identity(ns), v_t * np.identity(nt)


def count_saved(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def read_saved(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [float(line) for line in f if line.strip()]


def out_path(out_dir, setup, ns, method):
    return os.path.join(out_dir, f"FPR_pvalues_{setup}_ns{ns}_{method}.txt")


def run_task(task):
    si, ns, mi, reps, out_dir = task
    setup, dist, variance_mode = SETUPS[si]
    method, k_arg = METHODS[mi]
    path = out_path(out_dir, setup, ns, method)

    done = count_saved(path)
    if done < reps:
        with open(path, "a") as f:
            for rep in range(done, reps):
                # seed = (si + 1) * 1_000_000 + ns * 1000 + mi * 100 + rep
                # np.random.seed(seed)

                Xs, Xt, Ys, Yt = generate(
                    ns, NT, P, TRUE_BETA_S, TRUE_BETA_T, dist)
                Sigma_s, Sigma_t = build_sigma(
                    Xs, Xt, Ys, Yt, variance_mode)

                selected = DA_SeqFS.DA_SeqFS(
                    Xs, Ys, Xt, Yt, Sigma_s, Sigma_t,
                    k_arg, method="forward")
                j = np.random.randint(0, len(selected))
                pvalue = SI_SeqFS_DA.SI_SeqFS_DA(
                    Xs, Ys, Xt, Yt, k_arg, Sigma_s, Sigma_t,
                    method="forward", jth=j)

                f.write(f"{pvalue}\n")
                f.flush()
                print(f"[{setup:14s} ns={ns:3d} {method:14s}] "
                      f"rep {rep + 1}/{reps} p={pvalue}", flush=True)

    pvals = read_saved(path)
    fpr = sum(p <= ALPHA for p in pvals) / len(pvals) if pvals else float("nan")
    return setup, ns, method, fpr, len(pvals)


def resolve_setups(spec):
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(len(SETUPS)))
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in SETUP_NAMES]
    if unknown:
        raise ValueError(f"unknown setups: {unknown}; choices: {SETUP_NAMES}")
    return [SETUP_NAMES.index(name) for name in names]


def all_tasks(ns_list, reps, out_dir, setup_indices=None):
    if setup_indices is None:
        setup_indices = range(len(SETUPS))
    return [(si, ns, mi, reps, out_dir)
            for si in setup_indices
            for ns in ns_list
            for mi in range(len(METHODS))]


def run(workers, ns_list, reps, out_dir, setup_indices=None):
    os.makedirs(out_dir, exist_ok=True)
    tasks = all_tasks(ns_list, reps, out_dir, setup_indices)

    if workers <= 1:
        for task in tasks:
            setup, ns, method, fpr, n = run_task(task)
            print(f"[done] {setup:14s} ns={ns:3d} {method:14s} "
                  f"FPR={fpr:.4f} (n={n})", flush=True)
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            setup, ns, method, fpr, n = fut.result()
            print(f"[done] {setup:14s} ns={ns:3d} {method:14s} "
                  f"FPR={fpr:.4f} (n={n})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    parser.add_argument("--ns", type=str,
                        default=",".join(str(x) for x in NS_LIST))
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--setups", type=str, default="all",
                        help="comma-separated setups to run, or 'all' "
                             f"(choices: {','.join(SETUP_NAMES)})")
    args = parser.parse_args()
    run(args.workers, [int(x) for x in args.ns.split(",")],
        args.reps, args.outdir, resolve_setups(args.setups))
