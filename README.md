# SmartStore AI - Complete Fixed Version

## What is fixed
- Product category dropdown is present and populated.
- Flask explicitly sends categories to the prediction page.
- Fixed `TypeError: string indices must be integers`.
- CSV is loaded as a pandas DataFrame.
- Category demand is calculated from the selected category only.
- Monthly category records are aggregated by Bengaluru area before ranking.
- Top recommendation is automatically saved to History.
- Login/signup, Dashboard, History, Analytics and Admin are connected.
- Random Forest `.pkl` model is included.

## Run in VS Code / Windows

If PowerShell blocks `Activate.ps1`, don't activate the venv.

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe run.py
```




## Retrain model
```powershell
venv\Scripts\python.exe scripts\train_model.py
```

The included Bengaluru dataset is synthetic/demo data for academic use.
