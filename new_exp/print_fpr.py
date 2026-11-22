import argparse
import glob
import os


def read_pvalues(path):
    with open(path) as f:
        return [float(line) for line in f if line.strip()]


def compute_fpr(path, alpha):
    pvals = read_pvalues(path)
    n = len(pvals)
    fpr = sum(p <= alpha for p in pvals) / n if n else float("nan")
    return fpr, n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(here, "nonnormal")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=default_dir)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.txt")))
    if not paths:
        print(f"no .txt files found in {args.dir}")
        return

    width = max(len(os.path.basename(p)) for p in paths)
    print(f"{'file':<{width}}  {'n':>4}  {'FPR':>8}")
    print("-" * (width + 16))
    for path in paths:
        fpr, n = compute_fpr(path, args.alpha)
        print(f"{os.path.basename(path):<{width}}  {n:>4}  {fpr:>8.4f}")


if __name__ == "__main__":
    main()
