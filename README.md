# Botnet Traffic Detection System — CSIT375 Project

This project implements a supervised machine learning system for detecting botnet-related network traffic using the UNSW-NB15 cybersecurity dataset.

The system trains and evaluates multiple machine learning models, compares their performance, and provides a Tkinter-based graphical user interface for traffic classification, model visualization, confusion matrix analysis, class distribution viewing, and cross-validation.

This project was developed as part of CSIT375 Artificial Intelligence and Cybersecurity.

---

## Project Overview

Modern networks generate large volumes of traffic and log data, making manual analysis difficult and time-consuming. Botnet activity can also appear similar to normal traffic, especially when attackers use common protocols or attempt to hide command-and-control behavior.

This project demonstrates how machine learning can be used to classify network traffic as either benign or malicious by learning patterns from labelled network flow records.

The system focuses on binary classification:

- **Class 0:** Benign / normal traffic
- **Class 1:** Botnet / malicious traffic

---

## Features

- Machine-learning-based botnet traffic detection
- Random Forest, Logistic Regression, and K-Nearest Neighbors classifiers
- Tkinter-based GUI for user interaction
- Manual traffic classification through GUI input fields
- Confusion matrices displayed as text and heatmaps
- Model performance graphs
- Dataset class distribution visualization
- Stratified K-Fold cross-validation
- Consistent preprocessing between training and GUI prediction
- Stored encoders and scalers for reliable live prediction

---

## Model Performance Summary

The best-performing model was **Random Forest**.

| Model | Accuracy | Botnet Precision | Botnet Recall | Botnet F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 93.66% | 95% | 95% | 95% |
| K-Nearest Neighbors | 92.59% | 94% | 94% | 94% |
| Logistic Regression | 88.38% | 86% | 98% | 92% |

Random Forest was selected as the final operational model because it provided the best balance between overall accuracy, malicious traffic detection, and false positive control.

---

## Machine Learning Models

The system trains and evaluates the following models:

- **Random Forest**  
  Selected as the primary model for GUI-based traffic prediction due to its strong accuracy and balanced performance.

- **Logistic Regression**  
  Used as a baseline linear classifier for comparison.

- **K-Nearest Neighbors**  
  Used as a distance-based classifier to compare pattern similarity between traffic records.

---

## Dataset

The project uses the **UNSW-NB15** dataset, a cybersecurity dataset containing labelled network traffic records with both normal and attack traffic.

The dataset files used in this project are:

- `UNSW_NB15_training-set.xlsx`
- `UNSW_NB15_testing-set.xlsx`

The dataset includes network-flow features such as:

- connection duration,
- protocol,
- service,
- connection state,
- source bytes,
- destination bytes,
- packet rate,
- packet statistics,
- timing characteristics, and
- traffic labels.

---

## Dataset Processing Pipeline

The data pipeline includes:

1. Loading the UNSW-NB15 training and testing datasets.
2. Merging datasets for consistent preprocessing.
3. Selecting relevant network traffic features.
4. Handling missing or inconsistent values.
5. Encoding categorical features using stored encoders.
6. Scaling numerical features with `StandardScaler`.
7. Applying a 70/30 stratified train-test split.
8. Training and evaluating multiple classifiers.
9. Saving the selected model, encoder, and scaler.
10. Applying the same encoding and scaling pipeline inside the GUI during live prediction.

Using the same preprocessing pipeline for both training and GUI prediction ensures that input data is transformed consistently before classification.

---

## GUI Overview

The Tkinter interface provides several key functions.

### Prediction Graphs

Displays actual versus predicted class counts for each model.

### Confusion Matrices

Outputs confusion matrices as both text and heatmap visualizations.

### Live Traffic Prediction

Users manually input key network traffic features:

- `dur`
- `proto`
- `service`
- `state`
- `sbytes`
- `dbytes`
- `rate`

All remaining features required by the trained model are automatically assigned default values. The GUI then applies the same encoding and scaling pipeline used during training before making a prediction.

### Class Distribution

Displays the ratio of benign and malicious samples in the dataset.

### K-Fold Evaluation

Runs stratified K-Fold cross-validation on the selected model to evaluate model stability and generalization.

---

## Repository Contents

| File | Description |
|---|---|
| `GUI.py` | Graphical interface and user interaction logic |
| `Training.py` | Data preprocessing, encoding, scaling, model training, and evaluation |
| `UNSW_NB15_training-set.xlsx` | Training dataset |
| `UNSW_NB15_testing-set.xlsx` | Testing dataset |
| `README.md` | Project documentation |

> If your training file is named differently, update the table above to match the actual file name in the repository.

---

## Required Libraries

The project uses the following Python libraries and modules:

- `tkinter`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `time`
- `sys`

Important scikit-learn components used include:

- `train_test_split`
- `StratifiedKFold`
- `cross_val_score`
- `OrdinalEncoder`
- `LabelEncoder`
- `StandardScaler`
- `RandomForestClassifier`
- `LogisticRegression`
- `KNeighborsClassifier`
- `classification_report`
- `confusion_matrix`
- `accuracy_score`

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace the repository URL with the actual GitHub repository URL.

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

### 3. Make sure the following files are in the same directory

```text
GUI.py
Training.py
UNSW_NB15_training-set.xlsx
UNSW_NB15_testing-set.xlsx
```

### 4. Start the GUI

```bash
python GUI.py
```

---

## Example Use Cases

This project can be used to demonstrate:

- machine learning for cybersecurity,
- botnet traffic classification,
- network traffic analysis,
- model comparison,
- supervised classification workflows,
- GUI-based ML prediction, and
- basic security analytics visualization.

---

## My Role

Contributed to the development of the botnet traffic detection system as part of a group project.

My work included research, Python implementation, preprocessing logic, model training and evaluation, GUI functionality, testing, documentation, and result analysis.

The project involved building a supervised machine learning pipeline, comparing multiple classifiers, and integrating the selected model into a Tkinter-based interface for traffic classification and visualization.

---

## Ethical Use Notice

This project was developed for academic and defensive cybersecurity learning purposes.

It is intended to demonstrate how machine learning can support network traffic analysis and botnet detection using labelled datasets. It does not perform live exploitation, unauthorized monitoring, or offensive activity.

---

## Limitations and Future Work

### Current Limitations

- The system uses labelled historical dataset records rather than live network traffic.
- The GUI supports manual input-based traffic classification rather than real-time packet capture.
- The project uses supervised learning only.
- The current version does not include deep learning or unsupervised anomaly detection.
- Dataset imbalance and synthetic attack patterns may affect real-world generalization.

### Future Improvements

- Add live network traffic ingestion.
- Integrate SIEM or log pipeline support.
- Add additional anomaly detection models.
- Improve explainability using feature importance visualizations.
- Deploy the model as a lightweight API service.
- Add more advanced dashboards and reporting features.
- Improve GUI validation and usability.

---

## Notes

- Random Forest is the model used for final traffic prediction in the GUI.
- Missing or unknown inputs are handled using default values.
- The GUI uses the same encoding and scaling pipeline as the training phase to ensure consistent preprocessing.
- This project is developed for academic and research purposes under CSIT375 Artificial Intelligence and Cybersecurity.

---

## Project Status

Completed as an academic machine learning and cybersecurity project.

The system demonstrates how supervised learning can support automated botnet traffic detection and reduce the need for manual log review in large network environments.
