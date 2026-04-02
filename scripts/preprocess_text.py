"""
Unified text preprocessing pipeline.

Reads normalized JSONL files from jsonl_files_normalized/, combines them per
language, applies text cleaning (cleaned or verbatim mode), filters short
transcripts, performs stratified train/test splits, and writes output to
preprocessed_data/{mode}/.

Usage:
    python scripts/preprocess_text.py --mode cleaned
    python scripts/preprocess_text.py --mode verbatim
    python scripts/preprocess_text.py --mode both
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANG_CODE_MAP = {
    "eng": "en",
    "spa": "es",
    "zho": "zh",
    "ell": "el",
    "kor": "ko",
    "deu": "de",
}

MIN_CHAR_LENGTH = 60
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_config():
    config_path = os.path.join(project_root, "configs", "languages.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_to_lang(config):
    """Map dataset name -> language code (en, es, zh, etc.)."""
    mapping = {}
    for lang_code, lang_cfg in config["languages"].items():
        if lang_cfg.get("role") == "deferred":
            continue
        for ds in lang_cfg["datasets"]:
            if not ds.get("exclude", False):
                mapping[ds["name"]] = lang_code
    return mapping


def build_train_only_datasets(config):
    """Return set of dataset names that should only appear in training (no test split)."""
    train_only = set()
    for lang_code, lang_cfg in config["languages"].items():
        if lang_cfg.get("role") == "deferred":
            continue
        for ds in lang_cfg["datasets"]:
            if ds.get("train_only", False):
                train_only.add(ds["name"])
    return train_only


def load_all_normalized(input_dir, ds_to_lang, train_only_datasets):
    """Load all normalized JSONL files and tag each record with its language code."""
    records = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue
        ds_name = fname.replace(".jsonl", "")
        lang = ds_to_lang.get(ds_name)
        if lang is None:
            continue
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["lang"] = lang
                rec["_train_only"] = ds_name in train_only_datasets
                records.append(rec)
    return pd.DataFrame(records)


def clean_text_verbatim(text):
    """Verbatim mode: preserve disfluency markers, only strip timestamps and
    speaker labels that are artifacts of the extraction pipeline."""
    text = re.sub(r"\d{3,}_\d{3,}", "", text)
    text = re.sub(r"\bPAR:\s*", "", text)
    text = re.sub(r"\bINT:\s*", "", text)
    text = re.sub(r"\bPAR\d*:\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_cleaned(text):
    """Cleaned mode: aggressive cleaning matching the original MultiConAD pipeline.
    Strips disfluency markers, CHAT annotations, timestamps, punctuation artifacts."""
    text = re.sub(r"\d{3,}_\d{3,}", "", text)
    text = re.sub(r"\bPAR:\s*", "", text)
    text = re.sub(r"\bINT:\s*", "", text)
    text = re.sub(r"\bPAR\d*:\s*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"&[-=]\w+", "", text)
    text = re.sub(r"&\*\w+:\w+", "", text)
    text = re.sub(r"\(\.\.*\)", "", text)
    text = re.sub(r"xxx", "", text)
    text = re.sub(r"\bwww\b", "", text)
    text = text.replace("→", "")
    text = re.sub(r"[\\+^\"„]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df, mode, text_col="Text_participant"):
    """Apply text cleaning and filter short transcripts."""
    clean_fn = clean_text_verbatim if mode == "verbatim" else clean_text_cleaned

    df = df.copy()
    df["text_clean"] = df[text_col].fillna("").apply(clean_fn)
    df["text_len"] = df["text_clean"].apply(len)

    before = len(df)
    df = df[df["text_len"] >= MIN_CHAR_LENGTH].copy()
    dropped = before - len(df)
    if dropped:
        print(f"    Dropped {dropped} samples shorter than {MIN_CHAR_LENGTH} chars")

    return df


def split_and_save(df, lang, mode, output_base):
    """Stratified train/test split and save as JSONL.

    Records with _train_only=True are held out from the test split and
    added directly to training (matching the paper's WLS handling).
    """
    output_dir = os.path.join(output_base, mode)
    os.makedirs(output_dir, exist_ok=True)

    has_train_only = "_train_only" in df.columns and df["_train_only"].any()
    if has_train_only:
        df_splittable = df[~df["_train_only"]].copy()
        df_train_only = df[df["_train_only"]].copy()
        n_train_only = len(df_train_only)
    else:
        df_splittable = df.copy()
        df_train_only = pd.DataFrame()
        n_train_only = 0

    labels = df_splittable["Diagnosis"]
    label_counts = labels.value_counts()
    rare_labels = label_counts[label_counts < 2].index.tolist()
    if rare_labels:
        print(f"    WARNING: dropping {len(rare_labels)} classes with <2 samples: {rare_labels}")
        df_splittable = df_splittable[~df_splittable["Diagnosis"].isin(rare_labels)]
        labels = df_splittable["Diagnosis"]

    if len(df_splittable) < 5:
        print(f"    WARNING: only {len(df_splittable)} splittable samples for {lang}, no test split")
        train_df = df_splittable
        test_df = pd.DataFrame()
    else:
        train_df, test_df = train_test_split(
            df_splittable, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
        )

    if n_train_only > 0:
        train_df = pd.concat([train_df, df_train_only], ignore_index=True)

    cols_to_save = [
        "PID", "File_ID", "Dataset", "Languages", "lang", "Diagnosis",
        "Age", "Gender", "MMSE", "text_clean",
    ]
    existing_cols = [c for c in cols_to_save if c in df.columns]

    train_path = os.path.join(output_dir, f"train_{lang}.jsonl")
    train_df[existing_cols].to_json(train_path, orient="records", lines=True, force_ascii=False)

    if len(test_df) > 0:
        test_path = os.path.join(output_dir, f"test_{lang}.jsonl")
        test_df[existing_cols].to_json(test_path, orient="records", lines=True, force_ascii=False)
        train_only_note = f" (incl. {n_train_only} train-only)" if n_train_only else ""
        print(f"    train: {len(train_df)}{train_only_note}, test: {len(test_df)} -> {output_dir}/")
    else:
        print(f"    train only: {len(train_df)} -> {output_dir}/")


def process_mode(df_all, mode, output_base):
    """Process all languages for a given mode."""
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    for lang in sorted(df_all["lang"].unique()):
        df_lang = df_all[df_all["lang"] == lang]
        print(f"\n  [{lang}] {len(df_lang)} raw samples")

        diag_dist = df_lang["Diagnosis"].value_counts().to_dict()
        print(f"    Diagnosis: {diag_dist}")

        df_processed = preprocess_dataframe(df_lang, mode)
        split_and_save(df_processed, lang, mode, output_base)


def main():
    parser = argparse.ArgumentParser(description="Unified text preprocessing pipeline")
    parser.add_argument(
        "--mode", choices=["cleaned", "verbatim", "both"], default="both",
        help="Text cleaning mode"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Input directory (default: jsonl_files_normalized/)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: preprocessed_data/)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir or os.path.join(project_root, "jsonl_files_normalized")
    output_base = args.output_dir or os.path.join(project_root, "preprocessed_data")

    config = load_config()
    ds_to_lang = build_dataset_to_lang(config)
    train_only_datasets = build_train_only_datasets(config)

    print(f"Loading normalized JSONL from: {input_dir}")
    if train_only_datasets:
        print(f"Train-only datasets: {train_only_datasets}")
    df_all = load_all_normalized(input_dir, ds_to_lang, train_only_datasets)
    print(f"Total records loaded: {len(df_all)}")
    print(f"Languages: {sorted(df_all['lang'].unique())}")

    modes = ["cleaned", "verbatim"] if args.mode == "both" else [args.mode]
    for mode in modes:
        process_mode(df_all, mode, output_base)

    print(f"\nDone. Output: {output_base}/")


if __name__ == "__main__":
    main()
