Comparative Analysis of Machine Learning Models for Detection of Parkinson’s Disease from Audio Data

This project presents a comparative study of machine learning approaches for early detection of Parkinson’s disease using voice-based audio features. The system analyzes acoustic biomarkers extracted from speech recordings and evaluates multiple classification models to determine the most reliable method for disease prediction. Logistic Regression, Gradient Boosting, Random Forest, and K-Nearest Neighbors (KNN) are implemented and assessed under a unified experimental pipeline.

Features

Audio Feature Engineering:
Extract clinically relevant vocal features from speech recordings to capture neurological impairments associated with Parkinson’s disease.

Standardized Model Training:
Train multiple machine learning models using identical preprocessing, dataset splits, and evaluation criteria for fair comparison.

Performance Evaluation:
Assess model effectiveness using Accuracy, Precision, Recall, and F1-Score along with confusion matrix analysis.

Workflow Visualization:
Provide a structured flowchart illustrating preprocessing, training, evaluation, and result generation stages.

Project Structure
parkinsons-disease-detection/
├── codes/                     # Implementation of ML models
│   ├── logistic_regression.py
│   ├── gradient_boosting.py
│   ├── random_forest.py
│   └── knn.py
├── data/                      # Dataset directory
│   └── parkinsons_audio_data.csv
├── flowchart.png              # Methodology workflow diagram
├── README.md                  # Project documentation
└── results/                   # Generated outputs and visualizations
    ├── performance_metrics.csv
    └── model_comparison.png
Dataset

The dataset consists of labeled speech recordings categorized as Parkinson’s and healthy samples. Audio signals are transformed into structured numerical features including:

Jitter (frequency variation)

Shimmer (amplitude variation)

Harmonic-to-Noise Ratio (HNR)

Additional acoustic indicators of vocal stability and impairment

These features serve as biomarkers for detecting motor control degradation affecting speech production.

Models Used

Logistic Regression
Establishes a baseline linear classification model for binary disease prediction.

Gradient Boosting
Sequential ensemble learning method that improves predictive performance by minimizing classification errors iteratively.

Random Forest
Bagging-based ensemble approach leveraging multiple decision trees for improved robustness and generalization.

K-Nearest Neighbors (KNN)
Distance-based algorithm that classifies samples according to similarity in feature space.

Workflow

Preprocessing

Extract audio features using tools such as Librosa or Praat.

Normalize feature values.

Split dataset into training and testing subsets.

Training

Train all models on identical datasets.

Perform hyperparameter tuning to optimize performance.

Evaluation

Compare models using:

Accuracy

Precision

Recall

F1-Score

Generate confusion matrices for prediction analysis.

Results

Aggregate performance metrics in tabular format.

Visualize model comparison using graphical plots.

How to Run

Clone the repository:

git clone https://github.com/your-username/parkinson-s_disease.git
cd parkinson-s_disease

Install required dependencies:

pip install -r requirements.txt

Run individual model scripts:

python codes/lr.py
python codes/final.py
python codes/rf.py
python codes/knn.py
Project Goals

Compare classical machine learning algorithms for medical audio classification.

Identify the most reliable model for Parkinson’s detection.

Demonstrate an end-to-end ML pipeline from feature extraction to evaluation.

Provide a reproducible framework for healthcare-focused machine learning research
Flowchart:
![image](https://github.com/user-attachments/assets/095349c1-f6fd-44aa-acba-d920d07428cc)

