EEG Seizure Detection using RQA and Neural Networks
A machine learning approach to detecting epileptic seizures from EEG signals using Recurrence Quantification Analysis features and neural networks.
Results
The model achieves strong performance on held-out test data:

Accuracy: 93.4%
Precision: 94.7%
Recall: 87.8%
F1-Score: 0.911
AUC-ROC: 0.995

Overview
This project uses Recurrence Quantification Analysis (RQA) to extract meaningful features from EEG time series, then trains a feedforward neural network to classify seizure vs. non-seizure activity. The approach was validated on combined data from the CHB-MIT and Bonn EEG databases.
The key finding is that 5 carefully engineered RQA features outperform end-to-end deep learning (CNN) on this small medical dataset, likely because the features already capture the relevant temporal dynamics.
Dataset
The project combines two public EEG datasets:
CHB-MIT Scalp EEG Database

Pediatric patients with intractable seizures
256 Hz sampling rate
Contains annotated seizure events

Bonn EEG Database

Set A: Healthy subjects with eyes open
Set E: Epileptic seizure activity
173.61 Hz sampling rate

After preprocessing and augmentation, the combined dataset contains 527 samples (321 non-seizure, 206 seizure).
Methodology
Preprocessing
All signals were bandpass filtered (0.5-50 Hz) to remove noise outside the typical EEG frequency range. To address class imbalance, the minority seizure class was augmented using noise injection, amplitude scaling, and time-shifting techniques. Data was split 60/20/20 for training, validation, and testing.
Feature Extraction
Five RQA features were extracted from each EEG segment:

RR (Recurrence Rate): Density of recurrent states in the phase space
DET (Determinism): Measure of system predictability
L (Average Diagonal Line Length): Average prediction horizon
ENTR (Entropy): Complexity of the deterministic structure
LAM (Laminarity): Degree of intermittent behavior

These features capture nonlinear dynamics that distinguish seizure from non-seizure activity.
Model Architecture
Two models were compared:
Feedforward Neural Network (selected)

Three hidden layers: 64, 32, 16 neurons
Dropout regularization: 0.3, 0.3, 0.2
Validation F1: 0.988
Test F1: 0.911

1D Convolutional Neural Network (baseline)

Three convolutional layers with 64, 128, 256 filters
Validation F1: 0.766
Not selected due to overfitting

The simpler FFNN performed better because RQA features already encode the temporal patterns that CNNs would learn from raw signals.
Training
Models were trained with Adam optimizer (learning rate 0.001), binary cross-entropy loss, early stopping (patience 15), and learning rate reduction on plateau.
Installation
bashgit clone https://github.com/yourusername/eeg-seizure-detection.git
cd eeg-seizure-detection
pip install -r requirements.txt
Usage
Open and run the Jupyter notebook:
bashjupyter notebook eeg_seizure_detection.ipynb
The notebook walks through the complete pipeline: data loading, preprocessing, feature extraction, model training, and evaluation.
Requirements

Python 3.7+
NumPy 2.1.3
Pandas 2.2.3
TensorFlow 2.20.0
scikit-learn 1.6.1
MNE 1.11.0
SciPy 1.15.3
Matplotlib 3.10.0
Seaborn 0.13.2

See requirements.txt for the complete list with exact versions.
Limitations
This is a research project with several important limitations:

Small dataset (527 samples after augmentation)
Single-channel analysis only
Not validated in clinical settings
Not optimized for real-time detection
Trained on two specific datasets that may not generalize

This work is not intended for clinical use and should not be used for medical diagnosis.
Future Directions
Potential improvements include multi-channel analysis, real-time seizure prediction (detecting pre-ictal states), transfer learning from larger EEG datasets, explainability analysis using SHAP or attention mechanisms, and clinical validation studies.
References
Shoeb, A. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology.
Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6), 061907.
Marwan, N., Romano, MC, Thiel, M., Kurths, J. (2007). Recurrence plots for the analysis of complex systems. Physics Reports, 438(5-6), 237-329.
Data Access
The datasets used in this project are publicly available:

Bonn Database: http://epileptologie-bonn.de/cms/front_content.php?idcat=193
CHB-MIT Database: https://physionet.org/content/chbmit/1.0.0/

License
MIT License - see LICENSE file for details.
