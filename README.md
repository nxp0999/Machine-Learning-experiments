Spam Email Classification with Naive Bayes and Logistic Regression
Implemented and compared text classification algorithms for spam detection on Enron email datasets (enron1, enron2, enron4):
Algorithms:

Multinomial Naive Bayes (Bag-of-Words features)
Bernoulli Naive Bayes (binary presence/absence features)
Logistic Regression with L2 regularization (both feature representations)

Key Features:

Custom implementations from scratch (no sklearn/tensorflow)
Two feature representations: Bag-of-Words (word counts) and Bernoulli (binary)
Text preprocessing: lowercase conversion, punctuation removal, stopword filtering
Add-one Laplace smoothing for Naive Bayes
Gradient ascent optimization for Logistic Regression
Hyperparameter tuning using 70/30 train-validation split
Log-space probability calculations to prevent underflow

Evaluation:

Metrics: Accuracy, Precision, Recall, F1-Score
Comparative analysis across algorithms and feature representations
Performance reported separately for each dataset

Tech Stack: Python 3.9+, NumPy
