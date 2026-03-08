"""Compare experiment results from results/*.json files."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR


def load_results():
    """Load all result JSON files."""
    results = []
    if not os.path.exists(RESULTS_DIR):
        print("No results directory found.")
        return results

    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def print_comparison():
    results = load_results()
    if not results:
        print("No experiments found. Run experiment.py first.")
        return

    # Header
    header = f"{'Experiment':<30s} {'Model':>5s} {'Stg':>3s} {'#Feat':>5s} {'Norm':>4s} {'F1':>6s} {'Kappa':>6s} {'Subj F1':>8s} {'Time':>6s}"
    print(header)
    print('-' * len(header))

    # Sort by macro F1 descending
    results.sort(key=lambda r: r['report']['macro_f1'], reverse=True)

    for r in results:
        subj_mean = r['report']['per_subject_f1_mean']
        subj_std = r['report']['per_subject_f1_std']
        print(
            f"{r['name']:<30s} "
            f"{r['model'].upper():>5s} "
            f"{r['stages']:>3d} "
            f"{len(r['features']):>5d} "
            f"{'Y' if r['normalize'] else 'N':>4s} "
            f"{r['report']['macro_f1']:>6.4f} "
            f"{r['report']['kappa']:>6.4f} "
            f"{subj_mean:.2f}±{subj_std:.2f} "
            f"{r['elapsed_seconds']:>5.0f}s"
        )

    print(f"\n{len(results)} experiments total.")


if __name__ == '__main__':
    print_comparison()
