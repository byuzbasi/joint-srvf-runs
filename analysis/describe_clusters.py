"""Describe the held-out CAMUS subgroup in Table 7 from a completed run."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def summarize(run, dataset):
    manifest = json.loads((run / "manifest.json").read_text())
    output = run / manifest["attempt"] / "output"

    def read_table(name):
        data = (output / name).read_bytes()
        if [len(data), hashlib.sha256(data).hexdigest()] != manifest["outputs"][name]:
            raise ValueError(f"Archived table differs from manifest: {name}")
        return list(csv.DictReader(data.decode().splitlines()))

    splits = read_table("data/repeated_splits.csv")
    assignments = read_table("data/test_cluster_assignments.csv")
    test_ids = {row["patient_id"] for row in splits if row["split"] == "test"}
    if (len(splits) != 500 or len({r["patient_id"] for r in splits}) != 500
            or len(test_ids) != 150 or len(assignments) != 150
            or test_ids != {r["patient_id"] for r in assignments}):
        raise ValueError("Expected the archived 350/150 split and all 150 test assignments")

    groups = defaultdict(list)
    for row in assignments:
        ef = float(row["EF"])
        if ef < 50:
            continue
        path = dataset / row["patient_id"] / "Info_2CH.cfg"
        fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if (float(fields["EF"]) != ef or fields["Sex"] not in {"M", "F"}
                or fields["ImageQuality"] not in {"Good", "Medium", "Poor"}
                or (row["model"], row["k"]) != ("kmeans", "2")):
            raise ValueError(f"Inconsistent cluster or metadata for {row['patient_id']}")
        groups[row["cluster"]].append(fields)

    if set(groups) != {"0", "1"} or [len(groups[k]) for k in ("0", "1")] != [30, 22]:
        raise ValueError("Assignments differ from the cohort in Table 7")
    counts = {}
    summary = {}
    for k, rows in sorted(groups.items()):
        values = [float(r["EF"]) for r in rows]
        counts[k] = Counter(values)
        summary[k] = {
            "n": len(rows), "mean_EF": mean(values),
            "median_age_years": median(int(r["Age"]) for r in rows),
            "sex": dict(Counter(r["Sex"] for r in rows)),
            "two_chamber_quality": dict(Counter(r["ImageQuality"] for r in rows)),
        }
    shared = counts["0"].keys() & counts["1"].keys()
    return {
        "clusters": summary,
        "distinct_EF_values": len(counts["0"].keys() | counts["1"].keys()),
        "shared_EF_values": sorted(shared),
        "patients_at_shared_EF": sum(group[ef] for group in counts.values() for ef in shared),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed CAMUS run containing manifest.json")
    parser.add_argument("dataset", type=Path, help="CAMUS directory containing patient folders")
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.dataset), indent=2, sort_keys=True))
