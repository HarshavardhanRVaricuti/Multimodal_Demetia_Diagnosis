"""
Post-extraction label normalization: reads raw JSONL files from jsonl_files/,
applies a global label map + VAS-specific xlsx lookup, drops unlabeled samples,
and writes clean JSONL files to jsonl_files_normalized/.

Target taxonomy: HC, MCI, Dementia
"""
import os
import sys
import json

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_MAP = {
    # Healthy controls
    "Control": "HC",
    "Conrol": "HC",       # typo in Lu corpus
    "Healthy": "HC",
    "HC": "HC",
    # Dementia variants
    "ProbableAD": "Dementia",
    "PossibleAD": "Dementia",
    "Probable": "Dementia",
    "Alzheimer's": "Dementia",
    "Aphasia": "Dementia",
    "Pick's": "Dementia",
    "Vascular": "Dementia",
    "Dementia": "Dementia",
    "DTA": "Dementia",
    "AD": "Dementia",
    # MCI variants
    "Memory": "MCI",
    "ProbableMCI": "MCI",
    "PD-MCI": "MCI",
    "MCI": "MCI",
    # Drop
    "Other": "_exclude",
}

VAS_DIAG_MAP = {
    "H": "HC",
    "MCI": "MCI",
    "D": "Dementia",
}


def load_vas_lookup(data_root: str) -> dict:
    """Build VAS ID -> normalized diagnosis from 0demo.xlsx."""
    xlsx_path = os.path.join(data_root, "VAS", "0demo.xlsx")
    if not os.path.isfile(xlsx_path):
        print(f"  WARNING: VAS demo file not found at {xlsx_path}")
        return {}

    df = pd.read_excel(xlsx_path)
    lookup = {}
    for _, row in df.iterrows():
        vas_id = row.get("VAS ID")
        diag_raw = row.get("H/MCI/D")
        if pd.isna(vas_id) or pd.isna(diag_raw):
            continue
        vas_id_str = str(int(vas_id)).zfill(3)
        normalized = VAS_DIAG_MAP.get(str(diag_raw).strip())
        if normalized:
            lookup[vas_id_str] = normalized
    return lookup


def _wls_fluency_threshold(age):
    """Age-adjusted verbal fluency cutoff (MultiConAD paper Section 3.1.4)."""
    if age < 60:
        return 16
    elif age <= 79:
        return 14
    else:
        return 12


def load_wls_lookup(data_root: str) -> dict:
    """Build WLS file ID -> diagnosis using category fluency thresholds.

    Uses both 2011 and 2004 category fluency scored-word counts.
    If score in either year falls below the age-adjusted threshold -> Dementia.
    Otherwise -> HC.  Participants with no valid scores are excluded.
    """
    xlsx_path = os.path.join(data_root, "WLS", "WLS-data.xlsx")
    if not os.path.isfile(xlsx_path):
        print(f"  WARNING: WLS data file not found at {xlsx_path}")
        return {}

    df = pd.read_excel(xlsx_path)
    lookup = {}
    for _, row in df.iterrows():
        raw_id = int(row["idtlkbnk"])
        file_id = str(raw_id - 2000000000).zfill(5)

        age_2011 = row["age 2011"]
        cat_2011 = row["category fluency, scored words named, 2011"]
        thresh_2011 = _wls_fluency_threshold(age_2011)

        age_2004 = row["age 2004"]
        thresh_2004 = _wls_fluency_threshold(age_2004) if pd.notna(age_2004) and age_2004 > 0 else 16
        cat_2004 = row["category fluency, scored words named, 2004"]

        below_2011 = cat_2011 >= 0 and cat_2011 < thresh_2011
        below_2004 = pd.notna(cat_2004) and cat_2004 >= 0 and cat_2004 < thresh_2004
        has_valid = (cat_2011 >= 0) or (pd.notna(cat_2004) and cat_2004 >= 0)

        if not has_valid:
            continue

        lookup[file_id] = "Dementia" if (below_2011 or below_2004) else "HC"

    return lookup


def normalize_file(input_path: str, output_path: str, vas_lookup: dict,
                    wls_lookup: dict) -> dict:
    """Normalize labels in a single JSONL file. Returns count stats."""
    dataset_name = os.path.splitext(os.path.basename(input_path))[0]
    is_vas = dataset_name == "VAS"
    is_wls = dataset_name == "WLS"

    stats = {"kept": 0, "dropped_unknown": 0, "dropped_empty": 0, "dropped_exclude": 0}
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            raw_diag = record.get("Diagnosis", "Unknown")

            if is_vas:
                file_id = str(record.get("File_ID", "")).strip()
                normalized = vas_lookup.get(file_id)
                if normalized:
                    record["Diagnosis"] = normalized
                    records.append(record)
                    stats["kept"] += 1
                else:
                    stats["dropped_unknown"] += 1
                continue

            if is_wls:
                file_id = str(record.get("File_ID", "")).strip()
                normalized = wls_lookup.get(file_id)
                if normalized:
                    record["Diagnosis"] = normalized
                    records.append(record)
                    stats["kept"] += 1
                else:
                    stats["dropped_unknown"] += 1
                continue

            if not raw_diag or raw_diag.strip() == "":
                stats["dropped_empty"] += 1
                continue

            if raw_diag == "Unknown":
                stats["dropped_unknown"] += 1
                continue

            mapped = LABEL_MAP.get(raw_diag)
            if mapped == "_exclude":
                stats["dropped_exclude"] += 1
                continue

            if mapped is None:
                print(f"  WARNING: unmapped label '{raw_diag}' in {dataset_name}, dropping")
                stats["dropped_unknown"] += 1
                continue

            record["Diagnosis"] = mapped
            records.append(record)
            stats["kept"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    return stats


def main():
    import yaml

    config_path = os.path.join(project_root, "configs", "languages.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_root = os.path.expanduser(config["data_root"])

    input_dir = os.path.join(project_root, "jsonl_files")
    output_dir = os.path.join(project_root, "jsonl_files_normalized")
    os.makedirs(output_dir, exist_ok=True)

    vas_lookup = load_vas_lookup(data_root)
    print(f"VAS lookup loaded: {len(vas_lookup)} entries")

    wls_lookup = load_wls_lookup(data_root)
    print(f"WLS lookup loaded: {len(wls_lookup)} entries")

    total_kept = 0
    total_dropped = 0
    diag_counts = {}

    print(f"\n{'Dataset':25s} | {'Kept':>5s} | {'Dropped':>7s} | Distribution")
    print("-" * 80)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl"):
            continue

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        stats = normalize_file(input_path, output_path, vas_lookup, wls_lookup)

        dropped = stats["dropped_unknown"] + stats["dropped_empty"] + stats["dropped_exclude"]
        total_kept += stats["kept"]
        total_dropped += dropped

        if stats["kept"] > 0:
            with open(output_path, "r", encoding="utf-8") as f:
                ds_diag = {}
                for line in f:
                    d = json.loads(line).get("Diagnosis", "?")
                    ds_diag[d] = ds_diag.get(d, 0) + 1
                    diag_counts[d] = diag_counts.get(d, 0) + 1
                dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(ds_diag.items()))
        else:
            dist_str = "(empty)"
            os.remove(output_path)

        dataset = fname.replace(".jsonl", "")
        print(f"  {dataset:25s} | {stats['kept']:5d} | {dropped:7d} | {dist_str}")

    print(f"\n{'='*80}")
    print(f"Total kept: {total_kept}  |  Total dropped: {total_dropped}")
    print(f"\n--- Final Label Distribution ---")
    for k, v in sorted(diag_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
