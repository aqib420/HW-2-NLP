# HW-2-NLP
# NLP Homework - Fall 2024
Overview
This repository contains the solutions to Assignment 2 for the Natural Language Processing course, Fall 2024, under the guidance of Instructor Ayesha Enayet. The assignment focuses on implementing core NLP techniques, including:

N-Gram Language Modeling and Perplexity Calculation.
Naive Bayes Classifier for Sentiment Analysis.
Artificial Neural Network for Sentiment Classification.
Each task demonstrates foundational concepts in NLP and machine learning while emphasizing implementation from scratch.

Table of Contents
Overview
Technologies Used
Assignment Breakdown
Task 1: N-Gram Language Model
Task 2: Naive Bayes Classifier
Task 3: Artificial Neural Network
How to Run
Results
References
Technologies Used
Python 3.8 or higher
Libraries:
NumPy
Pandas
Matplotlib
tqdm
(No ML frameworks like TensorFlow or PyTorch for ANN)
Assignment Breakdown
Task 1: N-Gram Language Model
Objective: Implement an N-Gram language model and evaluate its performance using perplexity on a test dataset.
Key Features:
N-Gram probability calculation with add-one smoothing.
Comparison of perplexity for unigram, bigram, and trigram models.
Task 2: Naive Bayes Classifier
Objective: Perform binary sentiment classification on the IMDB dataset using a Naive Bayes classifier.
Key Features:
Tokenization, stop-word removal, and text preprocessing.
Prior probability and likelihood estimation with Laplace smoothing.
Metrics: Accuracy, confusion matrix, precision, recall, and F1-score.
Task 3: Artificial Neural Network
Objective: Implement an ANN for sentiment classification from scratch using one-hot encoded vectors.
Key Features:
Custom forward pass, backpropagation, and weight updates.
Metrics: Accuracy, confusion matrix, precision, recall, and F1-score.
How to Run
Clone the Repository:

bash
Copy code
git clone https://github.com/aqib420/HW-2-NLP.git
cd HW-2-NLP
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Individual Scripts:

Task 1 (N-Gram Model):
bash
Copy code
python ngram_model.py
Task 2 (Naive Bayes Classifier):
bash
Copy code
python naive_bayes.py
Task 3 (Artificial Neural Network):
bash
Copy code
python ann_sentiment.py
View Results: Check the results/ directory for generated output files, plots, and metrics.

Results
Task 1: N-Gram Language Model
Best Perplexity: Trigram model achieved the lowest perplexity on the test dataset.
Task 2: Naive Bayes Classifier
Accuracy: 85%
Precision/Recall/F1-Score:
Positive: Precision: 88%, Recall: 84%, F1-Score: 86%
Negative: Precision: 82%, Recall: 86%, F1-Score: 84%
Task 3: Artificial Neural Network
Accuracy: 80%
Confusion Matrix:
Predicted Positive	Predicted Negative
Actual Pos.	200	50
Actual Neg.	50	200
References
IMDB Dataset
Attention Is All You Need
Course Material: NLP Fall 2024
