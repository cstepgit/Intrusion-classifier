# Network Intrusion Classification

## Overview

This project compares XGBoost, random forest, and PyTorch neural-network classifiers for binary network-intrusion detection. Each notebook now uses the same leakage-resistant preprocessing and train/validation/test split implemented in `intrusion_classifier.py`.

## Setup

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Download `Train_data.csv` from the dataset below and place it at `Data/Train_data.csv`. The dataset is not committed to this repository.

Source: https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection?select=Train_data.csv

Run a notebook from the repository root:

```bash
jupyter notebook
```

The classifier notebooks:

- map `normal` to 0 and `anomaly` to 1;
- create stratified 64% training, 16% validation, and 20% test sets;
- fit one-hot encoding and numeric scaling only on the training set;
- reserve the test set for one final evaluation; and
- save fitted models and preprocessing artifacts under `artifacts/`.

## Corrected held-out results

These results were generated from the shared stratified split (`random_state=42`) on the labeled training file. The 5,039-row test partition was not used for preprocessing, model fitting, or early stopping.

| Model | Test accuracy | Anomaly precision | Anomaly recall | Anomaly F1 |
|---|---:|---:|---:|---:|
| XGBoost | 99.7023% | 99.8292% | 99.5317% | 99.6802% |
| Random forest | 99.5833% | 99.8288% | 99.2763% | 99.5518% |
| Neural network | 99.3451% | 99.4027% | 99.1911% | 99.2968% |

These held-out results measure performance on a random partition of `Train_data.csv`; they are not evidence of performance on unrelated or future network traffic.

## Why the scores are high

An audit found no duplicated labeled records crossing the training and test partitions. One identical feature vector appears in both partitions, but it has conflicting labels and therefore does not provide useful leakage.

The random test partition is nevertheless easy because it comes from the same source and contains the same attack patterns as the training data. Several categorical values are near-proxies for the binary label:

| Feature value | Samples | Anomaly rate |
|---|---:|---:|
| `flag=S0` | 7,009 | 98.86% |
| `service=private` | 4,351 | 95.43% |
| `service=domain_u` | 1,820 | 0.16% |
| `service=smtp` | 1,449 | 4.07% |
| `flag=SH` | 43 | 100.00% |
| `flag=RSTOS0` | 21 | 100.00% |

Many lower-frequency service values are 100% anomalous in this file. These are dataset-specific correlations, not general intrusion-detection rules. In the corrected XGBoost run, `service` accounts for 53.46% of aggregate feature importance.

Because the models were compared using these test results, this test partition must not be reused for further model selection. A defensible estimate of deployment performance requires evaluation on external labeled traffic, ideally from a different network or time period.

## Data

The dataset to be audited was provided which consists of a wide variety of intrusions simulated in a military network environment. It created an environment to acquire raw TCP/IP dump data for a network by simulating a typical US Air Force LAN. The LAN was focused like a real environment and blasted with multiple attacks. A connection is a sequence of TCP packets starting and ending at some time duration between which data flows to and from a source IP address to a target IP address under some well-defined protocol. Also, each connection is labelled as either normal or as an attack with exactly one specific attack type. Each connection record consists of about 100 bytes.
For each TCP/IP connection, 41 quantitative and qualitative features are obtained from normal and attack data (3 qualitative and 38 quantitative features) .The class variable has two categories:

• Normal

• Anomalous

![image](https://github.com/user-attachments/assets/6339a932-a333-4995-9921-86c80e164ad5)
![image](https://github.com/user-attachments/assets/1617ce42-87ad-4260-8e59-4c3cd2ebb1d4)
![image](https://github.com/user-attachments/assets/42649e18-5e62-470c-8a04-c0a2e2dc454a)





## XGBoost as the Primary Anomaly Detection Method

### Key Outcomes

### Model configuration

The XGBoost model uses 200 estimators, a learning rate of 0.1, and a maximum depth of 6. The results above replace historical metrics generated with the previous, leakage-prone workflow.

  ![image](https://github.com/user-attachments/assets/d73681ac-6ca1-4e4b-8b68-630a12dd3358)

  
- **Feature Importance Analysis:**  
  XGBoost highlighted several features as being highly predictive. The top 10 features identified were:
  - `service`
  - `src_bytes`
  - `wrong_fragment`
  - `logged_in`
  - `flag`
  - `hot`
  - `count`
  - `dst_host_same_srv_rate`
  - `dst_bytes`
  - `dst_host_same_src_port_rate`

  The notebook aggregates importance across one-hot encoded categories and selects the smallest source-feature set whose cumulative normalized importance is at least 95%. The exact set is computed from each new training run.

### Hypothesis

In the context of cybersecurity, the importance of these features can be hypothesized as follows:

- **`src_bytes` and `dst_bytes`:**  
  Abnormal data volumes transmitted from the source or to the destination may indicate data exfiltration attempts or denial-of-service attacks.

- **`protocol_type`:**  
  Certain protocols may be exploited more frequently by attackers; unusual or less common protocol usage in a network could raise red flags.

- **`hot` and `count`:**  
  These features often relate to the frequency and intensity of connections, potentially signaling scanning behavior or brute-force attempts.

- **Rate-based features (e.g., `diff_srv_rate`, `dst_host_same_src_port_rate`):**  
  Sudden changes in connection rates or patterns across services can indicate coordinated attacks or network probes.

These insights align with common attack patterns in cybersecurity, where deviations in network behavior—in volume, protocol, or connection patterns—are key indicators of malicious activity.

## Alternative Methodologies Explored

### Neural Network Approach (PyTorch)

- **Method:**  
  A deep neural network was built using a multi-layer perceptron architecture with early stopping based on validation loss to prevent overfitting.

- **Outcome:**  
  Although promising in performance, the neural network required extensive hyperparameter tuning and longer training times, making it less favorable compared to XGBoost.

The neural-network notebook uses a dedicated validation set for early stopping, restores the best validation checkpoint, and evaluates the test set once after training.

![image](https://github.com/user-attachments/assets/62cfbaf5-e6c6-46c2-9b15-647c4fb3ba79)
![image](https://github.com/user-attachments/assets/e62260e9-389c-4f38-b6c9-a97da247b77a)



### Random Forest Classifier

- **Method:**  
  A Random Forest model with 100 estimators was trained as an ensemble approach.

- **Outcome:**  
  Its corrected held-out metrics are reported in the comparison table above.

![image](https://github.com/user-attachments/assets/077d66ca-74ba-4261-bdaa-29d6a4d50cea)


## Conclusion

The notebooks provide reproducible model comparisons and persist the complete preprocessing required for inference. Model selection should be based on newly generated held-out test metrics rather than the historical screenshots retained above.
