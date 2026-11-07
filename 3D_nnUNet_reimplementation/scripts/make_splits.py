#!/usr/bin/env python3
"""Patient-grouped K-fold splits for nnU-Net v2, using train rows only."""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--preprocessed-dir",required=True)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    folder=Path(a.preprocessed_dir)
    if not (folder/"dataset_fingerprint.json").exists():
        raise ValueError("Run nnUNetv2_plan_and_preprocess before writing splits")
    if a.folds<2: raise ValueError("At least two folds required")
    groups=defaultdict(list)
    with Path(a.manifest).open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"].strip().lower()=="train":
                groups[r["patient_id"].strip()].append(r["case_id"].strip())
    if len(groups)<a.folds: raise ValueError("Not enough distinct training patients")
    cases={case for group in groups.values() for case in group}
    if len(cases)!=sum(map(len,groups.values())):
        raise ValueError("Duplicate train case IDs")
    patient_ids=list(groups)
    np.random.default_rng(a.seed).shuffle(patient_ids)
    patient_ids.sort(key=lambda v:len(groups[v]),reverse=True)
    groups_by_fold=[set() for _ in range(a.folds)]
    fold_sizes=[0]*a.folds
    for patient in patient_ids:
        k=min(range(a.folds),key=lambda x:(fold_sizes[x],len(groups_by_fold[x])))
        groups_by_fold[k].add(patient)
        fold_sizes[k]+=len(groups[patient])
    result=[]
    for group in groups_by_fold:
        val={case for patient in group for case in groups[patient]}
        result.append({"train":sorted(cases-val),"val":sorted(val)})
    if set().union(*(set(s["val"]) for s in result))!=cases:
        raise AssertionError("Split partition is incomplete")
    path=folder/"splits_final.json"
    if path.exists(): raise FileExistsError(f"Split already exists: {path}")
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(f"Wrote {len(result)} patient-grouped folds to {path}")
    print("Validation sizes:",[len(s["val"]) for s in result])
    print("No external data in cross-validation.")
if __name__=="__main__": main()
