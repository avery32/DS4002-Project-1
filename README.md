# Project 1: Hate Speech Identification 

## Section 1: Softwares and Platform 
The software used to complete this project was a google colab notebook. We utilized Python, specifically mainly Pandas and NumPy and the platforms used were Windows and Mac.  
The imports and libraries used:


## Section 2: Documentation Mapping 
In this section, you should provide an outline or tree illustrating the hierarchy of folders and subfolders contained in your Project Folder, and listing the files stored in each folder or subfolder.

## Section 3: Instructions 

**Assumptions:**  
- You are starting from the repository root.  
- Python ≥ 3.9 is installed.  
- GPU is optional but recommended for Transformer training.  

**Required Data:**  
- The raw CrowdFlower hate/offensive speech CSV saved as `DATA/labeled_data.csv`  
  (columns: `tweet`, `count`, `hate_speech`, `offensive_language`, `neither`, etc.)

---

### 3.1 Set up the environment
```bash
# (recommended) create & activate a virtual environment
python -m venv .venv

# Windows PowerShell
. .venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### 3.2 Place the raw data
Copy the original dataset into:
```
DATA/labeled_data.csv
```

### 3.3 Preprocess the data
```bash
python SCRIPTS/01_preprocess.py \
  --input DATA/labeled_data.csv \
  --drop-duplicates \
  --min-confidence 0.67
```
This writes:
```
DATA/processed/labeled_data_clean.csv
```
Optional flags:
* --min-confidence 0.67 → keep rows with ≥ 2/3 agreement
* --keep-all → keep all rows with count == 3

### 3.4 Train the Logistic Regression baseline
```bash
python SCRIPTS/02_model_logistic.py \
  --input DATA/processed/labeled_data_clean.csv \
  --out_dir OUTPUT \
  --test_size 0.20 \
  --seed 42
```

Artifacts produced in OUTPUT/:
* confusion_matrix_logistic.png
* logistic_best_params.json
* logistic_classification_report.txt
* logistic_metrics.json
* model_logistic.joblib

### 3.5 (Slow path) Fine-tune the Transformer (RoBERTa/BERT)
Tip: Use a GPU (e.g., Google Colab). CPU will work but is slow.

```bash
python SCRIPTS/03_model_transformer.py \
  --input DATA/processed/labeled_data_clean.csv \
  --out_dir OUTPUT_TRANSFORMER \
  --model_name roberta-base \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 128 \
  --seed 42
```

Artifacts produced in OUTPUT_TRANSFORMER/:
* confusion_matrix_transformer.png
* transformer_classification_report.txt
* transformer_metrics.json
* trainer/ (training logs/state)
* model/ (saved model + tokenizer)

### 3.6 (Fast Path) Run the Transformer via `DS4002_Project1_BERT.ipynb` on Google Colab

This path lets you skip environment setup locally and use a GPU in Colab. You’ll **upload the cleaned CSV** and the **Transformer script** so training starts immediately.

#### A) Open the notebook in Colab & enable GPU
1. Open `DS4002_Project1_BERT.ipynb` in Google Colab (File → Open Notebook → GitHub or upload).
2. Runtime → **Change runtime type** → **GPU** → Save.

#### B) Run the script
