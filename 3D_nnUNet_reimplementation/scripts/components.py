#!/usr/bin/env python3
"""Generate per-node connected-component candidates from 3D masks."""
import argparse,csv
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage

def extract(pred_dir,output_dir,min_volume_mm3=0.0):
    if min_volume_mm3<0:raise ValueError("min_volume_mm3 must be >= 0")
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for p in sorted(Path(pred_dir).glob("*.nii.gz")):
        img=nib.load(str(p))
        mask=np.asarray(img.dataobj)>0
        labs,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,3))
        voxel_vol=abs(np.linalg.det(img.affine[:3,:3]))
        mapped=np.zeros(mask.shape,dtype=np.int32)
        number=0
        for lab in range(1,n+1):
            inds=np.argwhere(labs==lab)
            volume=inds.shape[0]*voxel_vol
            if volume<min_volume_mm3:continue
            number+=1
            mapped[tuple(inds.T)]=number
            coords=nib.affines.apply_affine(img.affine,inds.mean(axis=0))
            low=inds.min(axis=0);high=inds.max(axis=0)
            records.append({
                "case_id":p.name[:-7],"candidate_id":number,
                "voxels":int(len(inds)),"volume_mm3":round(float(volume),4),
                "centroid_x_mm":round(float(coords[0]),3),
                "centroid_y_mm":round(float(coords[1]),3),
                "centroid_z_mm":round(float(coords[2]),3),
                "bbox_min_voxel":",".join(map(str,low)),
                "bbox_max_voxel":",".join(map(str,high))
            })
        volume_img=nib.Nifti1Image(mapped,img.affine,img.header.copy())
        volume_img.set_data_dtype(np.int32)
        nib.save(volume_img,str(out/p.name))
    fields=["case_id","candidate_id","voxels","volume_mm3",
            "centroid_x_mm","centroid_y_mm","centroid_z_mm",
            "bbox_min_voxel","bbox_max_voxel"]
    with (out/"ln_candidates.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    print(f"Extracted {len(records)} image components. "
          "These are not pathology-confirmed metastatic nodes.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--min-volume-mm3",type=float,default=0.0)
    a=p.parse_args()
    extract(a.pred_dir,a.output_dir,a.min_volume_mm3)
