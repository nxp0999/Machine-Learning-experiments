import numpy as np
import pandas as pd
from collections import defaultdict
import math

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for text classification (Bag of Words)
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier with Laplace smoothing parameter
        
        Args:
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
        self.n_features = None
        
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes classifier
        
        Args:
            X (numpy.ndarray): Training data (n_samples, n_features) - word counts
            y (numpy.ndarray): Training labels (n_samples,) - class labels
        """
        n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate class priors P(Y = c)
        for c in self.classes:
            class_count = np.sum(y == c)
            self.class_priors[c] = class_count / n_samples
        
        # Calculate feature probabilities P(X_j | Y = c)
        self.feature_probs = {}
        
        for c in self.classes:
            # Get samples of this class
            class_samples = X[y == c]
            
            # Sum word counts across all documents in this class
            total_word_counts = np.sum(class_samples, axis=0)
            
            # Total words in all documents of this class
            total_words = np.sum(total_word_counts)
            
            # Calculate probabilities with Laplace smoothing
            # P(X_j | Y = c) = (count(X_j, c) + alpha) / (sum_all_words(c) + alpha * n_features)
            feature_probs_c = (total_word_counts + self.alpha) / (total_words + self.alpha * self.n_features)
            
            self.feature_probs[c] = feature_probs_c
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for each class
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Log probabilities (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Start with log prior
            log_prior = math.log(self.class_priors[c])
            
            # Add log feature probabilities
            log_feature_probs = np.log(self.feature_probs[c])
            
            # For multinomial NB: sum over features of (count * log_prob)
            log_likelihood = np.dot(X, log_feature_probs)
            
            log_probs[:, i] = log_prior + log_likelihood
        
        return log_probs
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted class labels (n_samples,)
        """
        log_probs = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probs, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities (exponential of log probabilities)
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Class probabilities (n_samples, n_classes)
        """
        log_probs = self.predict_log_proba(X)
        
        # Convert log probabilities to probabilities using log-sum-exp trick for numerical stability
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return probs


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier for text classification (Binary features)
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize the classifier with Laplace smoothing parameter
        
        Args:
            alpha (float): Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None
        self.n_features = None
        
    def fit(self, X, y):
        """
        Train the Bernoulli Naive Bayes classifier
        
        Args:
            X (numpy.ndarray): Training data (n_samples, n_features) - binary features
            y (numpy.ndarray): Training labels (n_samples,) - class labels
        """
        n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate class priors P(Y = c)
        for c in self.classes:
            class_count = np.sum(y == c)
            self.class_priors[c] = class_count / n_samples
        
        # Calculate feature probabilities P(X_j = 1 | Y = c)
        self.feature_probs = {}
        
        for c in self.classes:
            # Get samples of this class
            class_samples = X[y == c]
            class_count = class_samples.shape[0]
            
            # Count how many documents in this class have each feature
            feature_counts = np.sum(class_samples, axis=0)
            
            # Calculate probabilities with Laplace smoothing
            # P(X_j = 1 | Y = c) = (count(X_j = 1, c) + alpha) / (count(c) + 2 * alpha)
            feature_probs_c = (feature_counts + self.alpha) / (class_count + 2 * self.alpha)
            
            self.feature_probs[c] = feature_probs_c
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for each class
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features) - binary features
            
        Returns:
            numpy.ndarray: Log probabilities (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Start with log prior
            log_prior = math.log(self.class_priors[c])
            
            # Get feature probabilities for this class
            feature_probs_c = self.feature_probs[c]
            
            # Calculate log probabilities for each sample
            log_likelihood = np.zeros(n_samples)
            
            for j in range(self.n_features):
                # For each feature, add log(P(X_j = x_j | Y = c))
                # If x_j = 1: log(P(X_j = 1 | Y = c))
                # If x_j = 0: log(P(X_j = 0 | Y = c)) = log(1 - P(X_j = 1 | Y = c))
                
                prob_1 = feature_probs_c[j]
                prob_0 = 1 - prob_1
                
                # Vectorized calculation
                mask_1 = X[:, j] == 1
                mask_0 = X[:, j] == 0
                
                log_likelihood[mask_1] += math.log(prob_1)
                log_likelihood[mask_0] += math.log(prob_0)
            
            log_probs[:, i] = log_prior + log_likelihood
        
        return log_probs
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted class labels (n_samples,)
        """
        log_probs = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probs, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Class probabilities (n_samples, n_classes)
        """
        log_probs = self.predict_log_proba(X)
        
        # Convert log probabilities to probabilities using log-sum-exp trick
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return probs


def load_dataset(file_path):
    """
    Load dataset from CSV file
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column (labels)
    
    return X, y


def test_naive_bayes():
    """
    Test both Naive Bayes implementations on the preprocessed data
    """
    datasets = ['enron1', 'enron2', 'enron4']
    results = {}
    
    for dataset in datasets:
        print(f"\n=== Testing {dataset.upper()} ===")
        
        # Test Multinomial Naive Bayes (Bag of Words)
        print("Multinomial Naive Bayes (Bag of Words):")
        try:
            # Load BoW data
            X_train, y_train = load_dataset(f'results/{dataset}_bow_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bow_test.csv')
            
            # Train classifier
            mnb = MultinomialNaiveBayes(alpha=1.0)
            mnb.fit(X_train, y_train)
            
            # Make predictions
            y_pred = mnb.predict(X_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            print(f"  Accuracy: {accuracy:.4f}")
            
            results[f'{dataset}_multinomial'] = {
                'predictions': y_pred,
                'true_labels': y_test,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Test Bernoulli Naive Bayes
        print("Bernoulli Naive Bayes (Binary features):")
        try:
            # Load Bernoulli data
            X_train, y_train = load_dataset(f'results/{dataset}_bernoulli_train.csv')
            X_test, y_test = load_dataset(f'results/{dataset}_bernoulli_test.csv')
            
            # Train classifier
            bnb = BernoulliNaiveBayes(alpha=1.0)
            bnb.fit(X_train, y_train)
            
            # Make predictions
            y_pred = bnb.predict(X_test)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            print(f"  Accuracy: {accuracy:.4f}")
            
            results[f'{dataset}_bernoulli'] = {
                'predictions': y_pred,
                'true_labels': y_test,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


if __name__ == "__main__":
    test_naive_bayes()