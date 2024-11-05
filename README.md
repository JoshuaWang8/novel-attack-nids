# Novel Attack Detection in Network Intrusion Detection Systems

## Author
Chiao-Hsi Joshua Wang

## Project Overview
This project explores the detection of novel cyberattacks in IoT networks using a multi-model approach that combines a binary classifier, a multi-class classifier, and a k-Nearest Neighbours (KNN) model. The goal is to develop a robust Network Intrusion Detection System (NIDS) capable of identifying known attack types while effectively detecting previously unseen threats. The methods and analysis presented here were developed as part of my Master of Data Science thesis at The University of Queensland.

Key highlights of the project include:

- Binary Classification: A neural network that classifies traffic as either benign or malicious.
- Multi-Class Classification: A neural network that identifies the specific attack type from a set of known classes.
- Novel Attack Detection: A KNN-based model that uses distance metrics to flag deviations from known traffic patterns as potential novel attacks.
- Voting Ensemble: An ensemble model that combines the predictions from the three sub-models to improve accuracy and enhance the detection of novel threats.

The project is built upon the UQ IoT IDS dataset, which includes samples of benign traffic and various types of network attacks. The code provided here covers feature extraction, data preprocessing and the model architecture. Some sample code for training the model and evaluating results is also included.

## Project Dependencies
The code in this repository requires the following dependencies:

- Python 3.8+
- PyTorch: For building and training neural network models.
- scikit-learn: For K-Nearest Neighbours and other machine learning utilities.
- pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations.
- scapy: For extracting features from raw network packets by the Kitsune feature extractor.

## Repository Overview
The repository is organized as follows:

- `/data`: Contains code for data feature extraction, sampling and preprocessing.
- `/models`: Contains implementation of the binary classifier, multi-class classifier, and KNN sub-models, as well as the voting ensemble. `training.ipynb` gives examples of how to use the model.
- `README.md`: This documentation file.

## Results

The models have been evaluated using accuracy and F1 scores, with promising results in detecting both known and novel attack types.

## Future Work

Potential areas for improvement include:
- Incorporating adaptive learning mechanisms to update the model in real-time as new attack types are identified.
- Exploring alternative machine learning algorithms for anomaly detection.
- Optimising the voting algorithm to further enhance detection accuracy.

## Acknowledgements
The code under the `/data/after_image/` directory was adapted from the Kitsune project, available at https://github.com/ymirsky/Kitsune-py.

I acknowledge Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai for their work, “Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection,” presented at the Network and Distributed System Security Symposium 2018 (NDSS’18).