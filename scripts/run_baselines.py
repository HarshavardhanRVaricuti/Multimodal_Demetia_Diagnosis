"""
Run the full MultiConAD baseline experiment matrix.

Experiment matrix (matching paper Tables 5 & 6):
  - 4 languages: en, es, zh, el
  - 2 tasks: binary (AD vs HC), ternary (AD vs MCI vs HC)
  - 2 settings: monolingual, combined-multilingual
  - 1 representation: sparse (TF-IDF)  [dense requires GPU]
  - 4 classifiers: DT, RF, SVM, LR

Total TF-IDF runs: 4 langs x 2 tasks x 2 settings = 16 runs (each runs 4 classifiers)

For dense (E5-large), set --include_dense (requires GPU or patience on CPU).

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --include_dense
    python scripts/run_baselines.py --languages en es --tasks binary
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAPER_LANGUAGES = ["en", "es", "zh", "el"]
TASKS = ["binary", "ternary"]


def run_experiment(script, test_lang, task, mode, train_languages, output_json):
    """Run a single experiment via subprocess."""
    cmd = [
        sys.executable, script,
        "--test_language", test_lang,
        "--task", task,
        "--mode", mode,
        "--output_json", output_json,
    ]
    if train_languages:
        cmd.extend(["--train_languages"] + train_languages)

    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return None

    print(result.stdout[-300:])
    return output_json


def build_summary_table(results_dir):
    """Build a summary table from all JSON result files."""
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)

        for clf_name, clf_results in data["results"].items():
            rows.append({
                "language": data["test_language"],
                "representation": data["representation"],
                "setting": data["setting"],
                "task": data["task"],
                "classifier": clf_name,
                "accuracy": clf_results["accuracy"],
            })

    if not rows:
        print("No results found.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    for task in TASKS:
        task_df = df[df["task"] == task]
        if task_df.empty:
            continue

        print(f"\n{'='*80}")
        print(f"{'Binary' if task == 'binary' else 'Multiclass'} Classification Results (Accuracy)")
        print(f"{'='*80}")

        for repr_name in ["sparse", "dense"]:
            repr_df = task_df[task_df["representation"] == repr_name]
            if repr_df.empty:
                continue

            print(f"\n--- {repr_name.upper()} ({('TF-IDF' if repr_name == 'sparse' else 'E5-large')}) ---")

            pivot = repr_df.pivot_table(
                index=["language", "setting"],
                columns="classifier",
                values="accuracy",
                aggfunc="first"
            )
            pivot = pivot.reindex(columns=["DT", "RF", "SVM", "LR"])
            print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser(description="Run MultiConAD baseline experiments")
    parser.add_argument("--languages", nargs="+", default=PAPER_LANGUAGES)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--mode", default="cleaned", choices=["cleaned", "verbatim"])
    parser.add_argument("--include_dense", action="store_true", help="Also run E5-large (slow without GPU)")
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or os.path.join(project_root, "results", f"baselines_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    tfidf_script = os.path.join(project_root, "Experiments", "TF_IDF_classifier.py")
    e5_script = os.path.join(project_root, "Experiments", "e5_larg_classifier.py")

    experiments = []

    for task in args.tasks:
        for test_lang in args.languages:
            # Monolingual
            experiments.append({
                "script": tfidf_script,
                "test_lang": test_lang,
                "task": task,
                "train_languages": None,
                "setting": "mono",
                "repr": "sparse",
            })
            # Combined-Multilingual
            experiments.append({
                "script": tfidf_script,
                "test_lang": test_lang,
                "task": task,
                "train_languages": args.languages,
                "setting": "multi",
                "repr": "sparse",
            })

            if args.include_dense:
                experiments.append({
                    "script": e5_script,
                    "test_lang": test_lang,
                    "task": task,
                    "train_languages": None,
                    "setting": "mono",
                    "repr": "dense",
                })
                experiments.append({
                    "script": e5_script,
                    "test_lang": test_lang,
                    "task": task,
                    "train_languages": args.languages,
                    "setting": "multi",
                    "repr": "dense",
                })

    print(f"Running {len(experiments)} experiments...")
    print(f"Results directory: {results_dir}")

    for i, exp in enumerate(experiments, 1):
        output_name = f"{exp['repr']}_{exp['setting']}_{exp['task']}_{exp['test_lang']}.json"
        output_json = os.path.join(results_dir, output_name)

        print(f"\n[{i}/{len(experiments)}] {exp['repr']} | {exp['setting']} | {exp['task']} | test={exp['test_lang']}")
        run_experiment(
            exp["script"], exp["test_lang"], exp["task"], args.mode,
            exp["train_languages"], output_json
        )

    print(f"\n\n{'#'*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'#'*80}")

    build_summary_table(results_dir)


if __name__ == "__main__":
    main()
