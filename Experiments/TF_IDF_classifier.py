"""
TF-IDF + sklearn classifier baseline (matches MultiConAD paper Section 4.2).

Classifiers: Decision Tree, Random Forest, SVM (linear+rbf), Logistic Regression
Grids: DT max_depth=[10,20,30], RF n_estimators=[50,100,200],
       SVM C=[0.1,1,10] kernel=[linear,rbf], LR C=[0.1,1,10]
Metric: accuracy (GridSearchCV scoring)

Usage:
    python Experiments/TF_IDF_classifier.py --test_language en --task binary --mode cleaned
    python Experiments/TF_IDF_classifier.py --test_language en --task ternary --mode cleaned --train_languages en es zh el
"""
import os
import sys
import argparse
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVAILABLE_LANGUAGES = ["en", "es", "zh", "el", "ko", "de"]
TEXT_COL = "text_clean"

CLASSIFIERS = {
    "DT": (DecisionTreeClassifier(random_state=42), {"max_depth": [10, 20, 30]}),
    "RF": (RandomForestClassifier(random_state=42), {"n_estimators": [50, 100, 200]}),
    "SVM": (SVC(random_state=42), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "LR": (LogisticRegression(random_state=42, max_iter=1000), {"C": [0.1, 1, 10]}),
}


def parse_args():
    parser = argparse.ArgumentParser(description="TF-IDF baseline classifier")
    parser.add_argument("--test_language", required=True, choices=AVAILABLE_LANGUAGES)
    parser.add_argument("--task", required=True, choices=["binary", "ternary"])
    parser.add_argument("--mode", default="cleaned", choices=["cleaned", "verbatim"])
    parser.add_argument(
        "--train_languages", nargs="+", default=None,
        help="Languages to include in training. Default: same as test (monolingual)."
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Base preprocessed data dir (default: preprocessed_data/)"
    )
    parser.add_argument("--output_json", default=None, help="Save results to JSON file")
    return parser.parse_args()


def load_split(data_dir, mode, split, lang):
    path = os.path.join(data_dir, mode, f"{split}_{lang}.jsonl")
    if not os.path.isfile(path):
        return None
    return pd.read_json(path, lines=True)


def run_classifiers(X_train, y_train, X_test, y_test, setting_info):
    """Run all 4 classifiers and return results dict."""
    results = {}
    for name, (clf_template, params) in CLASSIFIERS.items():
        grid_search = GridSearchCV(clf_template, params, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "best_params": grid_search.best_params_,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        print(f"  {name}: acc={acc:.4f} | params={grid_search.best_params_}")

    return results


def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(project_root, "preprocessed_data")
    train_langs = args.train_languages or [args.test_language]

    train_dfs = []
    for lang in train_langs:
        df = load_split(data_dir, args.mode, "train", lang)
        if df is not None:
            train_dfs.append(df)
            print(f"  Loaded train_{lang}: {len(df)} samples")

    if not train_dfs:
        print("ERROR: No training data found.")
        sys.exit(1)

    train_combined = pd.concat(train_dfs, ignore_index=True)

    test_df = load_split(data_dir, args.mode, "test", args.test_language)
    if test_df is None:
        print(f"ERROR: No test data for {args.test_language}")
        sys.exit(1)
    print(f"  Loaded test_{args.test_language}: {len(test_df)} samples")

    if args.task == "binary":
        train_combined = train_combined[train_combined["Diagnosis"] != "MCI"]
        test_df = test_df[test_df["Diagnosis"] != "MCI"]

    X_train_text = train_combined[TEXT_COL]
    y_train = train_combined["Diagnosis"]
    X_test_text = test_df[TEXT_COL]
    y_test = test_df["Diagnosis"]

    setting = "monolingual" if len(train_langs) == 1 else "combined-multilingual"
    print(f"\n{'='*60}")
    print(f"TF-IDF | {setting} | task={args.task} | test={args.test_language}")
    print(f"Train langs: {train_langs}")
    print(f"Train: {len(X_train_text)} | Test: {len(X_test_text)}")
    print(f"Train dist: {y_train.value_counts().to_dict()}")
    print(f"Test dist:  {y_test.value_counts().to_dict()}")
    print(f"{'='*60}")

    tfidf = TfidfVectorizer(sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    results = run_classifiers(X_train_tfidf, y_train, X_test_tfidf, y_test,
                              {"setting": setting, "task": args.task, "test_lang": args.test_language})

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        output = {
            "representation": "sparse",
            "setting": setting,
            "task": args.task,
            "test_language": args.test_language,
            "train_languages": train_langs,
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
