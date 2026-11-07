import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def run(script,*args):
    return subprocess.run([sys.executable,str(ROOT/"scripts"/script),
                           *map(str,args)],capture_output=True,text=True,check=True)

class WorkflowTests(unittest.TestCase):
    def test_synthetic_data_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);input_dir=d/"inputs";input_dir.mkdir()
            aff=np.diag([0.8,0.8,2.0,1.0]);rows=[]
            for i in range(5):
                ident=f"C{i:03d}"
                data=np.zeros((12,12,8),np.float32)
                data[2:6,3:6,1:4]=40+i
                label=(data>0).astype(np.uint8)
                ct=input_dir/f"{ident}_ct.nii.gz"
                mask=input_dir/f"{ident}_mask.nii.gz"
                nib.save(nib.Nifti1Image(data,aff),str(ct))
                nib.save(nib.Nifti1Image(label,aff),str(mask))
                rows.append({"case_id":ident,"patient_id":f"P{i:03d}",
                             "split":"train" if i<4 else "external_1",
                             "ct_path":str(ct),"mask_path":str(mask)})
            manifest=d/"cases.csv"
            with manifest.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            raw=d/"nnUNet_raw"
            run("prepare_dataset.py","--manifest",manifest,
                "--raw-root",raw,"--dataset-id",701)
            ds=raw/"Dataset701_CervicalLymphNodes"
            self.assertEqual(len(list((ds/"imagesTr").glob("*.nii.gz"))),4)
            self.assertEqual(len(list((ds/"labelsTr").glob("*.nii.gz"))),4)
            self.assertEqual(len(list((ds/"imagesTs").glob("*.nii.gz"))),1)
            self.assertFalse((ds/"labelsTr"/"C004.nii.gz").exists())
            self.assertEqual(json.loads((ds/"dataset.json").read_text())["numTraining"],4)
            prep=d/"preprocessed"/ds.name;prep.mkdir(parents=True)
            (prep/"dataset_fingerprint.json").write_text("{}")
            run("make_splits.py","--manifest",manifest,
                "--preprocessed-dir",prep,"--folds",2)
            folds=json.loads((prep/"splits_final.json").read_text())
            self.assertEqual(len(folds),2)
            self.assertEqual(set(folds[0]["val"]+folds[1]["val"]),
                             {"C000","C001","C002","C003"})
            self.assertNotIn("C004",set(folds[0]["train"]+folds[1]["train"]))
            pdir=d/"pred";rdir=d/"reference";pdir.mkdir();rdir.mkdir()
            for i in (0,1):
                name=f"C{i:03d}.nii.gz"
                src=ds/"labelsTr"/name
                (pdir/name).write_bytes(src.read_bytes())
                (rdir/name).write_bytes(src.read_bytes())
            result=d/"metrics.csv"
            run("evaluate.py","--pred-dir",pdir,
                "--ref-dir",rdir,"--output-csv",result)
            with result.open() as f:values=list(csv.DictReader(f))
            self.assertEqual(len(values),2)
            self.assertTrue(all(float(v["dice"])==1.0 for v in values))
            self.assertTrue(all(float(v["hd95_mm"])==0.0 for v in values))
            candidates=d/"candidates"
            run("components.py","--pred-dir",pdir,"--output-dir",candidates)
            with (candidates/"ln_candidates.csv").open() as f:
                lines=list(csv.DictReader(f))
            self.assertEqual(len(lines),2)

if __name__=="__main__":unittest.main()
