# Data

Do not commit generated split files or raw NEU images to this repository.

The NEU steel surface defect images should stay in a local dataset directory. Generate machine-local splits with:

```powershell
python scripts/prepare_dataset.py --dataset-dir "C:\Users\18747\Desktop\NEU surface defect database"
```

`data/splits.csv` contains absolute image paths, so it must be regenerated on each machine.
