"""
Batch extraction script: reads configs/languages.yaml and extracts all
datasets into jsonl_files/<dataset_name>.jsonl using CHACollection.
"""
import os
import sys
import json
import yaml
from dataclasses import asdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "Extracting data"))

from cha_collection import CHACollection
from collection import NormalizedDataPoint


def load_config():
    config_path = os.path.join(project_root, "configs", "languages.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_dataset(data_root, dataset_cfg, parser_language, output_dir):
    """Extract a single dataset into a JSONL file."""
    dataset_name = dataset_cfg["name"]
    rel_path = dataset_cfg["path"]
    dataset_path = os.path.join(data_root, rel_path)

    if dataset_cfg.get("exclude", False):
        print(f"  SKIP {dataset_name}: excluded in config")
        return 0

    if not os.path.isdir(dataset_path):
        print(f"  SKIP {dataset_name}: directory not found at {dataset_path}")
        return 0

    override_diagnosis = dataset_cfg.get("override_diagnosis", None)
    subfolder_label_map = dataset_cfg.get("subfolder_label_map", None)
    filename_label_map = dataset_cfg.get("filename_label_map", None)

    collection = CHACollection(
        path=dataset_path,
        language=parser_language,
        override_diagnosis=override_diagnosis,
        subfolder_label_map=subfolder_label_map,
        filename_label_map=filename_label_map,
    )

    output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
    count = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for normalized_datapoint in collection.get_normalized_data():
            normalized_dict = asdict(normalized_datapoint)
            json.dump(normalized_dict, outfile, ensure_ascii=False)
            outfile.write("\n")
            count += 1

    print(f"  {dataset_name}: {count} samples -> {output_path}")
    return count


def main():
    config = load_config()
    data_root = os.path.expanduser(config["data_root"])
    output_dir = os.path.join(project_root, "jsonl_files")
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for lang_code, lang_cfg in config["languages"].items():
        if lang_cfg.get("role") == "deferred":
            print(f"\n[{lang_code}] {lang_cfg['name']} -- DEFERRED, skipping")
            continue

        parser_language = lang_cfg["parser_language"]
        print(f"\n[{lang_code}] {lang_cfg['name']} (parser: {parser_language})")

        for dataset_cfg in lang_cfg["datasets"]:
            count = extract_dataset(data_root, dataset_cfg, parser_language, output_dir)
            total += count

    print(f"\n{'='*50}")
    print(f"Total samples extracted: {total}")
    print(f"Output directory: {output_dir}")

    # Print summary of diagnosis distribution
    print(f"\n--- Diagnosis Distribution ---")
    diagnosis_counts = {}
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(output_dir, fname)
        ds_counts = {}
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                diag = record.get("Diagnosis", "Unknown")
                ds_counts[diag] = ds_counts.get(diag, 0) + 1
        dataset = fname.replace(".jsonl", "")
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(ds_counts.items()))
        print(f"  {dataset:25s} | {dist_str}")
        for k, v in ds_counts.items():
            diagnosis_counts[k] = diagnosis_counts.get(k, 0) + v

    print(f"\n--- Overall ---")
    for k, v in sorted(diagnosis_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
