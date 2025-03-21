# Anomaly Detection in Network Traffic Using XGBoost - 99.84% Accuracy 

## Overview

Among the various models tested for anomaly detection in our network traffic data, the **XGBoost classifier** emerged as the best-performing method. Its gradient boosting framework not only achieved high accuracy but also delivered balanced precision and recall across classes. This approach has been adopted as our primary method, while alternative methodologies were also explored.

## Data 
Source: https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection?select=Train_data.csv

The dataset to be audited was provided which consists of a wide variety of intrusions simulated in a military network environment. It created an environment to acquire raw TCP/IP dump data for a network by simulating a typical US Air Force LAN. The LAN was focused like a real environment and blasted with multiple attacks. A connection is a sequence of TCP packets starting and ending at some time duration between which data flows to and from a source IP address to a target IP address under some well-defined protocol. Also, each connection is labelled as either normal or as an attack with exactly one specific attack type. Each connection record consists of about 100 bytes.
For each TCP/IP connection, 41 quantitative and qualitative features are obtained from normal and attack data (3 qualitative and 38 quantitative features) .The class variable has two categories:

• Normal

• Anomalous

![image](https://github.com/user-attachments/assets/6339a932-a333-4995-9921-86c80e164ad5)
![image](https://github.com/user-attachments/assets/1617ce42-87ad-4260-8e59-4c3cd2ebb1d4)
![image](https://github.com/user-attachments/assets/42649e18-5e62-470c-8a04-c0a2e2dc454a)





## XGBoost as the Primary Anomaly Detection Method

### Key Outcomes

### High Predictive Performance: Accuracy: 99.841% - F1-score: 1.00
  The XGBoost model, configured with 200 estimators, a learning rate of 0.1, and a maximum depth of 6, provided high accuracy and a favorable confusion matrix, effectively distinguishing between anomalous and normal events.
  ![image](https://github.com/user-attachments/assets/d73681ac-6ca1-4e4b-8b68-630a12dd3358)

  
- **Feature Importance Analysis:**  
  XGBoost highlighted several features as being highly predictive. The top 10 features identified were:
  - `src_bytes`
  - `protocol_type`
  - `hot`
  - `count`
  - `diff_srv_rate`
  - `dst_host_same_src_port_rate`
  - `dst_host_srv_count`
  - `dst_bytes`
  - `logged_in`
  - `dst_host_srv_diff_host_rate`

  A cumulative importance analysis further revealed that a set of 20 features accounted for 95% of the model's decision-making process. This insight facilitates targeted feature selection in future model iterations.

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

#### Resulting Accuracy: 99.404% - F1-score: 0.99
![image](https://github.com/user-attachments/assets/62cfbaf5-e6c6-46c2-9b15-647c4fb3ba79)
![image](https://github.com/user-attachments/assets/e62260e9-389c-4f38-b6c9-a97da247b77a)



### Random Forest Classifier

- **Method:**  
  A Random Forest model with 100 estimators was trained as an ensemble approach.

- **Outcome:**  
  This model delivered robust performance with a good bias-variance balance. However, it did not match the predictive power of the XGBoost model and lacked the detailed feature importance insights provided by XGBoost.

#### Resulting Accuracy: 99.781% - F1-score: 1.00
![image](https://github.com/user-attachments/assets/077d66ca-74ba-4261-bdaa-29d6a4d50cea)


## Conclusion

**XGBoost** has proven to be the most effective methodology for detecting anomalies in our network traffic dataset. Its strength lies in capturing complex feature interactions and providing clear interpretability through feature importance scores. The hypothesis—that features such as `src_bytes`, `protocol_type`, and various rate-based measures are key indicators of anomalous behavior—supports our findings and guides future improvements.

This comprehensive analysis lays the groundwork for deploying an efficient anomaly detection system that not only identifies potential threats but also offers critical interpretability for cybersecurity operations.
