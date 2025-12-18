Prerequisites

Python 3.8 or higher
Conda (recommended) or pip
Prokka (for genome annotation)
Abricate (for AMR gene detection)

Step 1: Clone Repository
bashgit clone https://github.com/yourusername/amr-prediction-streptococcus.git
cd amr-prediction-streptococcus
Step 2: Create Environment
Using Conda (Recommended):
bashconda env create -f environment.yml
conda activate amr-prediction
Using pip:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Step 3: Install Bioinformatics Tools
bash# Install Prokka
conda install -c bioconda prokka

# Install Abricate
conda install -c bioconda abricate

# Update Abricate databases
abricate --setupdb

📦 Requirements
Create a requirements.txt file:
txt# Core ML Libraries
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=2.0.0

# Data Processing
scipy>=1.10.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Bioinformatics
biopython>=1.80

# Utilities
joblib>=1.2.0
tqdm>=4.64.0

📊 Data Preparation
Step 1: Genome Download
Download Streptococcus mitis genomes from NCBI:
bash# Example using ncbi-genome-download
ncbi-genome-download bacteria \
  --genera "Streptococcus mitis" \
  --formats fasta \
  --output-folder data/raw/
Step 2: Genome Annotation
Annotate genomes using Prokka:
python# src/01_genome_annotation.py

import os
from pathlib import Path
import subprocess

genome_dir = Path("data/raw")
output_dir = Path("data/annotated")
output_dir.mkdir(exist_ok=True)

for genome_file in genome_dir.glob("*.fna"):
    strain_name = genome_file.stem
    out_path = output_dir / strain_name
    
    cmd = f"prokka --outdir {out_path} --prefix {strain_name} {genome_file}"
    subprocess.run(cmd, shell=True)
    
print("✔ Genome annotation completed!")
Step 3: Resistome Analysis
Detect AMR genes using Abricate:
python# src/02_resistome_analysis.py

import subprocess
import pandas as pd
from pathlib import Path

annotated_dir = Path("data/annotated")
output_file = "data/abricate_results.tsv"

# Run Abricate on all genomes
results = []
for genome_dir in annotated_dir.iterdir():
    if genome_dir.is_dir():
        fasta = genome_dir / f"{genome_dir.name}.fna"
        cmd = f"abricate --db resfinder {fasta}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        results.append(result.stdout)

# Combine results
with open(output_file, 'w') as f:
    f.write('\n'.join(results))

print(f"✔ Resistome analysis saved to {output_file}")
Step 4: Generate Feature Matrices
python# src/03_matrix_generation.py

import pandas as pd
import numpy as np
from pathlib import Path

# Load Abricate results
abricate_df = pd.read_csv("data/abricate_results.tsv", sep='\t')

# Create binary gene matrix
gene_matrix = pd.crosstab(
    abricate_df['FILE'],
    abricate_df['GENE'],
    values=1,
    aggfunc='max'
).fillna(0).astype(int)

# Create resistance class matrix
resistance_matrix = pd.crosstab(
    abricate_df['FILE'],
    abricate_df['RESISTANCE'],
    values=1,
    aggfunc='max'
).fillna(0).astype(int)

# Calculate Jaccard similarity matrices
def jaccard_similarity(matrix):
    intersection = matrix @ matrix.T
    union = matrix.sum(axis=1).values[:, None] + matrix.sum(axis=1).values - intersection
    return intersection / union

gene_jaccard = jaccard_similarity(gene_matrix)
res_jaccard = jaccard_similarity(resistance_matrix)

# Save matrices
output_dir = Path("data/matrices")
output_dir.mkdir(exist_ok=True)

gene_matrix.to_csv(output_dir / "abricate_resfinder_gene_matrix.csv")
resistance_matrix.to_csv(output_dir / "abricate_resfinder_resistance_matrix.csv")
pd.DataFrame(gene_jaccard, index=gene_matrix.index, columns=gene_matrix.index).to_csv(
    output_dir / "gene_gene_jaccard_matrix.csv"
)
pd.DataFrame(res_jaccard, index=resistance_matrix.index, columns=resistance_matrix.index).to_csv(
    output_dir / "res_res_jaccard_matrix.csv"
)

print("✔ Feature matrices generated successfully!")

🚀 Usage
Training the Model
Run the complete ML pipeline:
bashpython src/04_ml_pipeline.py
Or use the Python API:
pythonfrom src.ml_pipeline import AMRPipeline

# Initialize pipeline
pipeline = AMRPipeline(
    gene_matrix_path="data/matrices/abricate_resfinder_gene_matrix.csv",
    resistance_matrix_path="data/matrices/abricate_resfinder_resistance_matrix.csv",
    gene_jaccard_path="data/matrices/gene_gene_jaccard_matrix.csv",
    res_jaccard_path="data/matrices/res_res_jaccard_matrix.csv"
)

# Train models
pipeline.train()

# Evaluate
results = pipeline.evaluate()
print(results)

# Save models
pipeline.save_models("models/")
Making Predictions
Predict AMR for a new genome:
pythonfrom src.prediction import predict_amr

# Load models
gene_models = load_models("models/gene_models.pkl")
res_models = load_models("models/res_models.pkl")

# Prepare new genome features
new_genome_features = extract_features("new_genome.fna")

# Predict
gene_predictions = predict_genes(new_genome_features, gene_models)
resistance_predictions = predict_resistance(new_genome_features, res_models)

print("Predicted AMR Genes:", gene_predictions)
print("Predicted Antibiotic Resistance:", resistance_predictions)

🔄 Pipeline Workflow
mermaidgraph TD
    A[Raw Genomes] --> B[Prokka Annotation]
    B --> C[Abricate AMR Detection]
    C --> D[Binary Gene Matrix]
    C --> E[Resistance Matrix]
    D --> F[Jaccard Similarity]
    E --> F
    F --> G[PCA Feature Engineering]
    G --> H[Train-Test Split]
    H --> I[XGBoost Training]
    I --> J[Gene Prediction Model]
    I --> K[Resistance Prediction Model]
    J --> L[Evaluation & Metrics]
    K --> L
    L --> M[Model Deployment]
Detailed Steps

Data Collection: Download S. mitis genomes from NCBI
Annotation: Predict genes using Prokka
AMR Detection: Identify resistance genes with Abricate + ResFinder
Feature Engineering:

Binary presence/absence matrices
Jaccard similarity calculations
PCA dimensionality reduction (5 components each)


Data Cleaning:

Handle missing values
Remove zero-variance features
Align strain indices across matrices


Train-Test Split: 75/25 split with random seed
Model Training: XGBoost classifiers for each label
Evaluation: Comprehensive metrics and reports
Model Saving: Pickle serialization for deployment


📈 Results
Model Performance
Gene Prediction (Multi-Label)
MetricScoreInterpretationMicro F10.90+Excellent overall gene predictionMacro F10.55-0.70Good balance across rare/common genesJaccard Index0.80+High overlap between predicted and true gene sets
Antibiotic Resistance Prediction
Antibiotic ClassPrecisionRecallF1-ScoreSupportTetracycline0.920.890.9045Macrolide0.880.850.8638Aminoglycoside0.850.820.8332Beta-lactam0.780.750.7628
Feature Importance
Top 10 most important features for AMR prediction:

tet(M)_2 - Tetracycline resistance
erm(B)_1 - Macrolide resistance
mef(A)_2 - Macrolide efflux
GENE_PC1 - Primary gene similarity component
aph(3')-III_1 - Aminoglycoside resistance

Sample Visualizations
(Add your actual figures here)
python# Generate confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# For a specific antibiotic
cm = confusion_matrix(y_test['tetracycline'], predictions['tetracycline'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Tetracycline Resistance Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/figures/tetracycline_confusion_matrix.png')

🔬 Model Architecture
XGBoost Hyperparameters
pythonxgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42
}
Feature Engineering
Input Features:

Binary gene presence/absence (50-100 genes)
PCA embeddings from gene Jaccard matrix (5 components)
PCA embeddings from resistance Jaccard matrix (5 components)
Optional: Strain metadata (species, phylogenetic cluster)

Total Feature Dimensions: ~60-110 features per strain

🧪 Reproducibility
Random Seeds
All random processes use seed 42:

Train-test split
XGBoost model initialization
PCA initialization

Environment
bash# Export exact environment
conda env export > environment_exact.yml

# Freeze pip requirements
pip freeze > requirements_exact.txt

📚 Citation
If you use this code in your research, please cite:
bibtex@article{your_paper_2024,
  title={Machine Learning Prediction of Antimicrobial Resistance in Streptococcus mitis},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX},
  doi={10.xxxx/xxxxx}
}

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
txt
