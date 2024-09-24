# Intrusion Detection System: Comparative Analysis Using CICIoT2023 and SWaT Datasets

## Overview
This repository contains a comparative analysis of different machine learning techniques applied to the **CICIoT2023** and **SWaT** datasets for building an intrusion detection system (IDS). The models used in this study include Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN). Performance of these models is evaluated based on accuracy, precision, recall, F1-score, and false positives (FP).

## Datasets
- **CICIoT2023**: A dataset for IoT network traffic classification, where the task is to identify different types of network attacks.
- **SWaT (Secure Water Treatment)**: A dataset focused on detecting anomalies in water treatment processes.

## Techniques and Models
The following machine learning techniques were applied:

1. **Logistic Regression**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**

Each model was evaluated for its ability to classify network attacks and normal traffic.

## Results and Findings

### 1. CICIoT2023 Dataset

#### **Technique 1: Logistic Regression**
Logistic Regression was applied to classify 7 different attack types. The model’s performance was evaluated using precision, recall, F1-score, and confusion matrices for both multi-class (8 classes including benign traffic) and binary classification (2 classes: attack vs benign).

- **Classification Report (8 Classes)**: Logistic regression achieved high performance for DDos and Mirai attacks, but struggled with Brute Force, Dos, Recon, Spoofing, and Web attacks, yielding a relatively high false positive (FP) rate of 246.
  
- **Classification Report (2 Classes)**: Performance improved significantly in binary classification (attack vs benign).

#### **Technique 2: Random Forest**
The Random Forest model significantly improved classification accuracy.

- **Classification Report (8 Classes)**: Achieved a remarkable accuracy of 99%. Despite this, the FP rate was 250, slightly higher than logistic regression.
  
- **Classification Report (2 Classes)**: Showed high precision, recall, and F1-score for both classes.


#### **Conclusion for CICIoT2023 Dataset**
- **Logistic Regression**: Offers an accuracy of 83%, with some limitations in classifying certain attack types.
- **Random Forest**: Offers superior performance, achieving an accuracy of 99%, with a marginally higher FP rate. Given the importance of minimizing false positives in intrusion detection systems, Random Forest’s higher accuracy makes it a better choice overall.

### 2. SWaT Dataset

#### **Technique 1: Logistic Regression**
The logistic regression model was applied to detect anomalies in the SWaT dataset.

- **Classification Report**: Showed good accuracy (95%) but a relatively high FP rate (7395).
  
- **Confusion Matrix**: Highlight areas where the model struggled.

#### **Technique 2: Random Forest**
The Random Forest model was tested on the SWaT dataset, yielding the best performance metrics:

- **Classification Report**: Achieved the highest accuracy (97%) with an FP rate of 3167.
  
- **Confusion Matrix**: Figures 11 and 12 show improved performance over Logistic Regression.

#### **Technique 3: K-Nearest Neighbors (KNN)**
The KNN model was also applied to the SWaT dataset.

- **Classification Report**: Achieved an accuracy of 94%, with a lower FP rate (2878) than Logistic Regression.
  
- **Confusion Matrix**: Demonstrate KNN's moderate performance.

#### **Conclusion for SWaT Dataset**
- **Logistic Regression**: Offers decent performance with 95% accuracy, but struggles with high false positive rates.
- **KNN**: Provides a lower FP rate compared to Logistic Regression but falls short in accuracy.
- **Random Forest**: Delivers the best overall performance with 97% accuracy and a balanced trade-off between precision, recall, and FP rate.

### Summary of Results

| Dataset         | Model               | Accuracy | False Positives (FP) |
|-----------------|---------------------|----------|----------------------|
| CICIoT2023      | Logistic Regression  | 83%      | 246                  |
| CICIoT2023      | Random Forest        | 99%      | 250                  |
| SWaT            | Logistic Regression  | 95%      | 7395                 |
| SWaT            | KNN                  | 94%      | 2878                 |
| SWaT            | Random Forest        | 97%      | 3167                 |

## Conclusion
Across both datasets, the **Random Forest** model consistently outperformed other techniques, offering the highest accuracy and balanced precision/recall for both attack and benign classes. Although its FP rate is slightly higher in some cases, its overall performance makes it the most suitable model for an intrusion detection system. 

## Installation and Usage

### Requirements:
- Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib
