# S01 Self Directed Activity

This repository contains an analysis and implementation of machine learning models for the Iris dataset, serving as a summary of previously learned concepts.

## Repository Structure

### 📁 datasets/

- `breast+cancer+wisconsin+diagnostic/`: Contains breast cancer dataset files
  - `wdbc.data`: Raw data
  - `wdbc.names`: Dataset description
- `iris/`: Contains iris dataset files
  - `iris.data`: Raw data
  - `iris.names`: Dataset description
  - `bezdekIris.data`: Alternative version
  - `Index`: Index file
- `iris_data_splits/`: Contains train/dev/test splits
  - `x_train.csv`, `y_train.csv`: Training data
  - `x_dev.csv`, `y_dev.csv`: Development data
  - `x_test.csv`, `y_test.csv`: Test data
- `iris_normalized_data.csv`: Normalized iris dataset

### 📁 models/

- `mlp.joblib`: Saved MLP model
- `optimized_random_forest.joblib`: Saved optimized Random Forest model

### 📓 Notebooks

#### N01_EDA.ipynb

An exploratory data analysis of both the Iris and Breast Cancer Wisconsin datasets. Includes:

- Data loading and preprocessing
- Statistical summaries
- Data visualization
- Feature analysis
- Data normalization

#### N02_models.ipynb

Implementation and comparison of 3 classification models for the Iris dataset:

- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- Model evaluation and comparison using various metrics
- Hyperparameter tuning
- Model persistence for later comparison

#### N03_MLP_Comparison.ipynb

Implementation of a Multi-Layer Perceptron (MLP) classifier and comparison with previous models:

- MLP implementation using scikit-learn
- Model training and evaluation
- Performance comparison with Random Forest
- Final model selection

## Requirements

- Python 3.10.16
- Key libraries: pandas, numpy, scikit-learn, matplotlib, plotly
- Can be found on requirements.txt

```bash
pip install -r requirements.txt
```
