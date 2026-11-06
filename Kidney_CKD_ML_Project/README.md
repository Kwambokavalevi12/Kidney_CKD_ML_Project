# Early Kidney Failure / CKD Classification — Data Mining Project

**Group:** Data Mining Group  
**Members:** Bentley Busienei (SBENBU2211), Valentine Kwamboka (SKWAOM2421), Brighton Wanyama (SNABBR2321)

## Project purpose
A supervised-learning educational prototype that groups adult NHANES participants into:
1. No kidney abnormality detected by the selected one-visit criteria
2. Early kidney abnormality
3. Advanced kidney impairment

This is a research classification exercise, not a medical diagnosis or clinical decision-support product.

## Dataset
Official CDC/NCHS NHANES August 2021–August 2023 public-use files. The Python program downloads the XPT files directly from:
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023

Files used: DEMO_L, BIOPRO_L, ALB_CR_L, BMX_L, BPXO_L, DIQ_L, SMQ_L, PAQ_L, ALQ_L, GHB_L and BPQ_L. The final eligible sample size is printed and saved after successful execution; it is not assumed in advance.

## Method and leakage prevention
The 2021 CKD-EPI creatinine equation estimates eGFR for research grouping. Urine albumin-to-creatinine ratio (uACR) is also used to define the outcome. Serum creatinine, eGFR and uACR are excluded from predictors to prevent target leakage. Predictors include age, sex, BMI, blood pressure, diabetes, HbA1c/glucose, smoking and available lifestyle variables.

For this classroom project, eGFR <45 is assigned to advanced impairment; eGFR 45–59 or eGFR >=60 with uACR >=30 mg/g is assigned to early kidney abnormality; eGFR >=60 and uACR <30 mg/g is assigned to no abnormality detected. These are cross-sectional survey categories only. A single survey visit cannot establish persistence for at least three months, which is part of the clinical definition of CKD.

## Run in VS Code (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/ and tick **Add Python to PATH**.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run the model:
   ```powershell
   python src/run_project.py
   ```
The first run needs an internet connection to download the public CDC files. It may take several minutes. If the downloads fail, try again on a stable connection.

## Outputs
After a successful run, inspect:
- `results/model_comparison.csv`
- `results/classification_report.csv`
- `results/run_summary.json`
- `results/confusion_matrix.png`
- `results/model_comparison.png`
- `results/feature_importance.png`
- `models/best_ckd_classifier.joblib`

Use the actual values in these files in the Results and Discussion section. Do not invent or estimate scores.

## Evaluation
The script uses an 80:20 stratified train/test split and reports accuracy, macro precision, macro recall, macro F1 and macro one-vs-rest ROC-AUC where calculable. The best model is selected by test-set macro F1 for this demonstration; for a formal study, model selection should be done within cross-validation on the training data and final testing should occur only once.

## Limitations
- NHANES is a cross-sectional survey; persistence of kidney abnormalities cannot be confirmed.
- Some questionnaire and laboratory variables have missing observations.
- The simple classroom pipeline does not fully implement NHANES complex survey-weighted inference.
- The model is not externally validated and must not be used to diagnose or treat anyone.
- Results apply to the sampled U.S. survey population and may not generalize to Kenya or other populations.

## Ethical use
Use public de-identified data only. Do not upload private patient data. Do not use the classifier to make individual medical decisions.
