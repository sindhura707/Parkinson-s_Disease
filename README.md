# Comparative-Analysis-of-ML-models-for-detection-of-Parkinson-s-Disease-from-audios.
This project focuses on detecting Parkinson's disease using machine learning models trained on audio features. The study compares the performance of Logistic Regression, Gradient Boosting, Random Forest, and K-Nearest Neighbors (KNN) classifiers to determine the most effective approach for accurate detection.

Features
Preprocessing Audio Data: Extract relevant features from audio recordings.
Training and Evaluation: Compare four ML models on the same dataset.
Performance Metrics: Analyze models based on accuracy, precision, recall, and F1-score.
Flowchart Visualization: A detailed flowchart to outline the process.
Project Structure
plaintext
Copy code
parkinsons-disease-detection/  
├── codes/                            # Folder containing code for each ML model  
│   ├── logistic_regression.py  
│   ├── gradient_boosting.py  
│   ├── random_forest.py  
│   ├── knn.py  
├── data/                             # Dataset folder (add your audio dataset here)  
│   ├── parkinsons_audio_data.csv  
├── flowchart.png                     # Flowchart illustrating the methodology  
├── README.md                         # Project documentation  
└── results/                          # Folder for saving model results and graphs  
    ├── performance_metrics.csv  
    ├── model_comparison.png  
Dataset
The dataset contains audio recordings labeled as Parkinson's and non-Parkinson's.
Audio features include jitter, shimmer, HNR (Harmonic-to-Noise Ratio), and other parameters essential for detecting vocal impairment.
Models Used
Logistic Regression: A simple yet effective classifier for binary problems.
Gradient Boosting: An ensemble method that optimizes classification accuracy through iterative boosting.
Random Forest: A robust ensemble learning method using decision trees.
K-Nearest Neighbors (KNN): A non-parametric algorithm based on proximity.
Workflow
1. Preprocessing
Extract features from the audio recordings using libraries such as Librosa or Praat.
Normalize and split the dataset into training and testing sets.

3. Training
Train each model using the same dataset and perform hyperparameter tuning for optimal performance.

4. Evaluation:   
Compare models based on performance metrics such as:
Accuracy,
Precision,
Recall.
F1-Score,
Use a confusion matrix to visualize predictions.

5. Results
Summarize results in a tabular format for easy comparison.
Visualize performance metrics using bar graphs.
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/parkinsons-disease-detection.git  
cd parkinsons-disease-detection  
Install the necessary libraries:

Run the individual model scripts:

bash
python codes/lr.py  
python codes/final.py  
python codes/rf.py  
python codes/knn.py  

Flowchart:
![image](https://github.com/user-attachments/assets/095349c1-f6fd-44aa-acba-d920d07428cc)
