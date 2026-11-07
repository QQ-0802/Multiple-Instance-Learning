#!/usr/bin/env python3
"""Prepare binary cervical-LN NIfTI masks for official nnU-Net v2."""
import argparse, csv, json, re, shutil
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np

def parse_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        fields={"case_id","patient_id","split","ct_path","mask_path"}
        if not fields.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest needs columns {sorted(fields)}")
        return list(reader)

def image(path):
    if not path.is_file() or not str(path).endswith(".nii.gz"):
        raise ValueError(f"Missing .nii.gz file: {path}")
    img=nib.load(str(path))
    if len(img.shape)!=3 or not np.all(np.isfinite(img.affine)):
        raise ValueError(f"Invalid 3D NIfTI geometry: {path}")
    return img

def source(path, manifest):
    p=Path(path).expanduser()
    return p if p.is_absolute() else manifest.parent/p

def validated_mask(ct, mask_path):
    seg=image(mask_path)
    if seg.shape!=ct.shape or not np.allclose(seg.affine,ct.affine,atol=1e-3,rtol=0):
        raise ValueError(f"CT and mask do not share shape/affine: {mask_path}")
    arr=np.asarray(seg.dataobj)
    if not np.isfinite(arr).all() or not np.isin(np.unique(arr),[0,1]).all():
        raise ValueError(f"Segmentation must contain binary 0/1 labels: {mask_path}")
    return seg, arr

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--raw-root",required=True)
    p.add_argument("--dataset-id",type=int,required=True)
    p.add_argument("--dataset-name",default="CervicalLymphNodes")
    p.add_argument("--clip-low",type=float)
    p.add_argument("--clip-high",type=float)
    a=p.parse_args()
    if not (0<=a.dataset_id<=999): raise ValueError("dataset-id must be 0..999")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",a.dataset_name):
        raise ValueError("Invalid dataset-name")
    if (a.clip_low is None)!=(a.clip_high is None):
        raise ValueError("Specify both clip-low and clip-high, or neither")
    if a.clip_low is not None and a.clip_low>=a.clip_high:
        raise ValueError("clip-low must be smaller than clip-high")
    manifest=Path(a.manifest).resolve()
    rows=parse_manifest(manifest)
    if not rows: raise ValueError("Manifest is empty")
    counts=Counter(); ids=set(); patients={}
    for r in rows:
        case=r["case_id"].strip()
        patient=r["patient_id"].strip()
        split=r["split"].strip().lower()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",case):
            raise ValueError(f"Invalid pseudonymous case ID: {case}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",patient):
            raise ValueError(f"Invalid pseudonymous patient ID: {patient}")
        if case in ids: raise ValueError(f"Duplicate case: {case}")
        if split not in ("train","external_1","external_2"):
            raise ValueError(f"Invalid split: {split}")
        if patient in patients and patients[patient]!=split:
            raise ValueError(f"Patient in two cohorts: {patient}")
        if split=="train" and not (r["mask_path"] or "").strip():
            raise ValueError(f"Missing training mask: {case}")
        patients[patient]=split; ids.add(case); counts[split]+=1
    if counts["train"]==0: raise ValueError("No training cases")
    target=Path(a.raw_root).resolve()/f"Dataset{a.dataset_id:03d}_{a.dataset_name}"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")
    for folder in ("imagesTr","labelsTr","imagesTs"):
        (target/folder).mkdir(parents=True,exist_ok=True)
    for r in rows:
        case=r["case_id"].strip()
        split=r["split"].strip().lower()
        ct_path=source(r["ct_path"],manifest)
        ct=image(ct_path)
        dest=target/("imagesTr" if split=="train" else "imagesTs")/f"{case}_0000.nii.gz"
        if a.clip_low is None:
            shutil.copy2(ct_path,dest)
        else:
            arr=np.asarray(ct.dataobj,dtype=np.float32)
            if not np.isfinite(arr).all(): raise ValueError(f"Non-finite CT values: {ct_path}")
            out=nib.Nifti1Image(np.clip(arr,a.clip_low,a.clip_high),ct.affine,ct.header.copy())
            out.set_data_dtype(np.float32); nib.save(out,str(dest))
        mask_str=(r["mask_path"] or "").strip()
        if mask_str:
            seg,arr=validated_mask(ct,source(mask_str,manifest))
            if split=="train":
                out=nib.Nifti1Image(arr.astype(np.uint8),seg.affine,seg.header.copy())
                out.set_data_dtype(np.uint8)
                nib.save(out,str(target/"labelsTr"/f"{case}.nii.gz"))
            # External labels are verified but never copied to training labels.
    dataset_json={
        "channel_names":{"0":"CT"},
        "labels":{"background":0,"lymph_node":1},
        "numTraining":counts["train"],
        "file_ending":".nii.gz"
    }
    (target/"dataset.json").write_text(json.dumps(dataset_json,indent=2)+"\n")
    print(f"Created {target}; training={counts['train']}, "
          f"external1={counts['external_1']}, external2={counts['external_2']}")
    print("External masks excluded from training labels.")
if __name__=="__main__": main()
