#!/usr/bin/env python3
"""Evaluate segmentation with binary Dice and symmetric 95th percentile HD (mm)."""
import argparse, csv
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

def read_mask(path):
    img=nib.load(str(path))
    if len(img.shape)!=3: raise ValueError(f"Not 3D: {path}")
    values=np.asarray(img.dataobj)
    if not np.isin(np.unique(values),[0,1]).all():
        raise ValueError(f"Expected 0/1 mask: {path}")
    return img,values>0

def dice(a,b):
    denom=int(a.sum())+int(b.sum())
    return 1.0 if denom==0 else 2.0*np.count_nonzero(a&b)/denom

def hd95(a,b,spacing):
    if not a.any() and not b.any():return 0.0
    if not a.any() or not b.any():return float("nan")
    struct=ndimage.generate_binary_structure(3,1)
    sa=a&~ndimage.binary_erosion(a,structure=struct,border_value=0)
    sb=b&~ndimage.binary_erosion(b,structure=struct,border_value=0)
    da=ndimage.distance_transform_edt(~sa,sampling=spacing)
    db=ndimage.distance_transform_edt(~sb,sampling=spacing)
    return float(np.percentile(np.concatenate((db[sa],da[sb])),95))

def evaluate(pred_dir,ref_dir,output_csv):
    predictions=sorted(Path(pred_dir).glob("*.nii.gz"))
    if not predictions:raise ValueError("Prediction folder contains no NIfTI masks")
    records=[]
    for p in predictions:
        ref=Path(ref_dir)/p.name
        if not ref.exists():raise FileNotFoundError(f"No reference mask: {ref}")
        pi,pa=read_mask(p); ri,ra=read_mask(ref)
        if pa.shape!=ra.shape or not np.allclose(pi.affine,ri.affine,atol=1e-3,rtol=0):
            raise ValueError(f"Prediction/reference geometry mismatch: {p.name}")
        spacing=nib.affines.voxel_sizes(ri.affine)
        records.append({"case_id":p.name[:-7],"dice":round(float(dice(pa,ra)),6),
                        "hd95_mm":round(float(hd95(pa,ra,spacing)),6),
                        "pred_voxels":int(pa.sum()),"reference_voxels":int(ra.sum())})
    out=Path(output_csv);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    ds=np.array([r["dice"] for r in records])
    hs=np.array([r["hd95_mm"] for r in records])
    hsummary=f"{np.mean(hs[np.isfinite(hs)]):.3f}" if np.isfinite(hs).any() else "undefined"
    print(f"Cases={len(records)}; mean Dice={ds.mean():.4f}; "
          f"median Dice={np.median(ds):.4f}; mean defined HD95={hsummary} mm")
    print(f"Per-case results: {out}")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir",required=True)
    parser.add_argument("--ref-dir",required=True)
    parser.add_argument("--output-csv",required=True)
    args=parser.parse_args()
    evaluate(args.pred_dir,args.ref_dir,args.output_csv)
