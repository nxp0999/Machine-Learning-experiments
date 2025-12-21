import numpy as np
import pandas as pd

class LogisticRegression:
   
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization_lambda=0.0, verbose=False):
       
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization_lambda = regularization_lambda
        self.verbose = verbose
        self.weights = None
        self.cost_history = []
        
    def sigmoid(self, z):
       
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
       
        n_samples, n_features = X.shape
        
        # Augment X with column of ones for bias term
        X_augmented = np.hstack([np.ones((n_samples, 1)), X])
        
        # Initialize weights (bias is weights[0], features are weights[1:])
        self.weights = np.random.randn(n_features + 1) * 0.01
        
        # Gradient ascent
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X_augmented @ self.weights
            predictions = self.sigmoid(z)
            
            # Calculate cost (negative log-likelihood with L2 regularization)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            
            log_likelihood = np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Add L2 regularization (only on feature weights, NOT bias)
            l2_penalty = 0
            if self.regularization_lambda > 0:
                l2_penalty = (self.regularization_lambda / (2 * n_samples)) * np.sum(self.weights[1:] ** 2)
            
            cost = -log_likelihood + l2_penalty
            self.cost_history.append(cost)
            
            # Calculate gradients
            error = y - predictions
            gradient = (X_augmented.T @ error) / n_samples
            
            # Apply L2 regularization to feature weights only (not bias at weights[0])
            if self.regularization_lambda > 0:
                gradient[1:] -= (self.regularization_lambda / n_samples) * self.weights[1:]
            
            # Gradient ascent update
            self.weights += self.learning_rate * gradient
            
            # Print progress
            if self.verbose and iteration % 100 == 0:
                print(f"  Iteration {iteration}/{self.max_iterations}, Cost: {cost:.6f}")
    
    def predict_proba(self, X):
        
        X_augmented = np.hstack([np.ones((X.shape[0], 1)), X])
        z = X_augmented @ self.weights
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def train_test_split(X, y, test_size=0.3, random_state=42, stratify=None):
    
    np.random.seed(random_state)
    n_samples = len(X)
    
    if stratify is not None:
        # Stratified split to maintain class proportions
        unique_classes = np.unique(stratify)
        train_indices = []
        test_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            n_cls_test = int(len(cls_indices) * test_size)
            
            # Shuffle indices for this class
            np.random.shuffle(cls_indices)
            
            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
    else:
        # Simple random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def load_dataset(file_path):
   
    df = pd.read_csv(file_path)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column (labels)
    
    return X, y


def tune_hyperparameters(X_train, y_train, X_val, y_val, verbose=False):
  
    
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    lambdas = [0.0, 0.001, 0.01, 0.1]
    
    best_accuracy = 0
    best_params = {}
    
    for lr in learning_rates:
        for lam in lambdas:
            # Train model
            model = LogisticRegression(
                learning_rate=lr,
                max_iterations=1000,
                regularization_lambda=lam,
                verbose=False  # Suppress individual model training output
            )
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            
            if verbose:
                print(f"    LR={lr}, Lambda={lam}: Val Acc={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'learning_rate': lr,
                    'regularization_lambda': lam,
                    'validation_accuracy': accuracy
                }
    
    
    return best_params


def test_logistic_regression():
   
    datasets = ['enron1', 'enron2', 'enron4']
    representations = ['bow', 'bernoulli']
    results = {}
    
    
    print("LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING")
    
    
    for dataset in datasets:
        print(f"\n=== Testing {dataset.upper()} ===")
        
        for rep in representations:
            print(f"\nLogistic Regression ({rep.upper()} features):")
            
            try:
                # Load data
                X_train, y_train = load_dataset(f'results/{dataset}_{rep}_train.csv')
                X_test, y_test = load_dataset(f'results/{dataset}_{rep}_test.csv')
                
                # Split training data for hyperparameter tuning (70% train, 30% val)
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
                )
                
                # Tune hyperparameters
                best_params = tune_hyperparameters(
                    X_train_split, y_train_split, 
                    X_val_split, y_val_split,
                    verbose=False
                )
                
                # Train final model on full training set with best parameters
                print("  Training final model on full training set...")
                final_model = LogisticRegression(
                    learning_rate=best_params['learning_rate'],
                    max_iterations=2000,  # More iterations for final model
                    regularization_lambda=best_params['regularization_lambda'],
                    verbose=True
                )
                final_model.fit(X_train, y_train)
                
                # Make predictions on test set
                y_pred = final_model.predict(X_test)
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y_test)
                print(f"  Test Accuracy: {accuracy:.4f}")
                
                results[f'{dataset}_{rep}'] = {
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'accuracy': accuracy,
                    'best_params': best_params
                }
                
            except Exception as e:
                print(f"  Error: {e}")
    
    return results


if __name__ == "__main__":
    # Test logistic regression
    results = test_logistic_regression()
    
    # Summary
    
    print("LOGISTIC REGRESSION SUMMARY")
    
    print(f"\n{'Dataset_Representation':<25} {'Accuracy':<12} {'Best LR':<12} {'Best Lambda':<12}")

    
    for key, result in results.items():
        best_params = result['best_params']
        print(f"{key:<25} {result['accuracy']:<12.4f} "
              f"{best_params['learning_rate']:<12.4f} "
              f"{best_params['regularization_lambda']:<12.4f}")
    
    # Calculate averages
    bow_accs = [v['accuracy'] for k, v in results.items() if 'bow' in k]
    bern_accs = [v['accuracy'] for k, v in results.items() if 'bernoulli' in k]
    
    if bow_accs:
        print(f"\nAverage BoW accuracy: {np.mean(bow_accs):.4f}")
    if bern_accs:
        print(f"Average Bernoulli accuracy: {np.mean(bern_accs):.4f}")