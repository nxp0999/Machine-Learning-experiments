# CS 6375 Project 1: Spam Email Classification
## Naive Bayes and Logistic Regression Implementation

### Project Overview
This project implements three machine learning algorithms from scratch for spam email classification:
- **Multinomial Naive Bayes** (for word count features)
- **Bernoulli Naive Bayes** (for binary presence/absence features)
- **Logistic Regression with L2 Regularization** (for both feature types)

The implementation uses only NumPy for core algorithms (no scikit-learn or TensorFlow), achieving O(nd) complexity per iteration for efficient training.

---

## Project Structure

```
PROJECT1/
├── data/                          # Enron email datasets
│   ├── enron1_train/train/
│   │   ├── spam/
│   │   └── ham/
│   ├── enron1_test/test/
│   │   ├── spam/
│   │   └── ham/
│   ├── enron2_train/train/
│   │   ├── spam/
│   │   └── ham/
│   ├── enron2_test/test/
│   │   ├── spam/
│   │   └── ham/
│   ├── enron4_train/train/
│   │   ├── spam/
│   │   └── ham/
│   └── enron4_test/test/
│       ├── spam/
│       └── ham/
├── results/                       # Generated CSV files (created automatically)
│   ├── enron1_bow_train.csv
│   ├── enron1_bow_test.csv
│   ├── enron1_bernoulli_train.csv
│   ├── enron1_bernoulli_test.csv
│   ├── enron2_bow_train.csv
│   ├── enron2_bow_test.csv
│   ├── enron2_bernoulli_train.csv
│   ├── enron2_bernoulli_test.csv
│   ├── enron4_bow_train.csv
│   ├── enron4_bow_test.csv
│   ├── enron4_bernoulli_train.csv
│   └── enron4_bernoulli_test.csv
├── src/                           # Source code files
│   ├── __pycache__/              # Python cache (auto-generated)
│   ├── data_preprocessing.py     # Text preprocessing and feature extraction
│   ├── naive_bayes.py           # Naive Bayes implementations
│   ├── logistic_regression.py   # Logistic Regression with L2 regularization
│   ├── evaluation.py            # Metrics calculation and reporting
│   └── main.py                  # Main orchestration script
└── README.md                     # This file
```

---

## Requirements

### Python Version
- Python 3.9 or later

### Dependencies
```bash
pip install numpy pandas nltk
```

### Download NLTK Stopwords (first time only)
```python
python -c "import nltk; nltk.download('stopwords')"
```

---

## Dataset Information

### Enron Email Datasets
All three datasets are included with the following structure:

- **enron1:** ~450 training, ~456 test emails
- **enron2:** ~463 training, ~478 test emails  
- **enron4:** ~535 training, ~543 test emails

### Directory Structure
Each dataset follows this pattern:
```
enronX_train/train/spam/    # Training spam emails (.txt files)
enronX_train/train/ham/     # Training ham emails (.txt files)
enronX_test/test/spam/      # Test spam emails (.txt files)
enronX_test/test/ham/       # Test ham emails (.txt files)
```

The `data_preprocessing.py` script automatically handles this nested path structure.

---

## Running the Project

### Option 1: Run Complete Pipeline (Recommended)
This runs preprocessing, trains all models, and displays comprehensive results:

```bash
python src/main.py
```

**What it does:**
1. Checks if data is preprocessed (creates feature matrices if needed)
2. Trains Multinomial NB, Bernoulli NB, and Logistic Regression on all datasets
3. Evaluates all models with accuracy, precision, recall, and F1-score
4. Performs hyperparameter tuning for Logistic Regression
5. Displays comparative analysis and best hyperparameters
6. Shows overall winner

---

## How main.py Orchestrates Everything

`main.py` is the **central orchestration script** that runs the complete project pipeline from start to finish. It ties all components together:

### What main.py Does:

1. **Automatic Data Preparation**
   - Checks if preprocessed CSV files exist in `results/`
   - If missing, automatically calls `DatasetLoader` from `data_preprocessing.py` to create them
   - Ensures feature matrices are ready before training

2. **Model Training Pipeline**
   - Imports and instantiates all three algorithms:
     - `MultinomialNaiveBayes` and `BernoulliNaiveBayes` from `naive_bayes.py`
     - `LogisticRegression` from `logistic_regression.py`
   - Trains each model on appropriate feature types (BoW or Bernoulli)
   - Trains across all three datasets (enron1, enron2, enron4)

3. **Hyperparameter Tuning** (for Logistic Regression)
   - Grid search over learning rates: [0.01, 0.05, 0.1, 0.5]
   - Grid search over lambda values: [0.0, 0.001, 0.01, 0.1]
   - Finds optimal parameters for each dataset and feature type

4. **Comprehensive Evaluation**
   - Uses `calculate_metrics()` from `evaluation.py` for accuracy, precision, recall, F1
   - Calls `print_metrics_table()` to display formatted results
   - Calls `print_comparison_analysis()` to show best algorithms per dataset

5. **Final Summary**
   - Displays overall average accuracies for all algorithms
   - Shows best performing algorithm
   - Reports optimal hyperparameters found

### Workflow Comparison:

**Running main.py (Recommended):**
```bash
python src/main.py
```
→ Complete automated pipeline: preprocessing → training → tuning → evaluation → results

**Running individual components (For debugging/testing):**
```bash
python src/data_preprocessing.py    # Only create feature matrices
python src/naive_bayes.py           # Only test Naive Bayes models
python src/logistic_regression.py   # Only test Logistic Regression
python src/evaluation.py            # Only evaluate pre-trained models
```

**Summary:** Use `main.py` for final results and complete project execution. Use individual scripts for component-level testing and debugging.

---

### Option 2: Run Individual Components

#### Step 1: Data Preprocessing Only
Converts raw emails into feature matrices (BoW and Bernoulli):

```bash
python src/data_preprocessing.py
```

**Outputs:** 12 CSV files in `results/` directory:
- `enron1_bow_train.csv`, `enron1_bow_test.csv`
- `enron1_bernoulli_train.csv`, `enron1_bernoulli_test.csv`
- (Same for enron2 and enron4)

---

#### Step 2: Test Naive Bayes Models
Trains and evaluates both NB variants:

```bash
python src/naive_bayes.py
```

**Output:** Accuracy results for Multinomial and Bernoulli NB on all datasets

---

#### Step 3: Test Logistic Regression
Trains and evaluates LR with both feature types:

```bash
python src/logistic_regression.py
```

**Output:** Accuracy results with training progress (cost decreasing over iterations)

---

#### Step 4: Comprehensive Evaluation
Runs all models and displays detailed metrics:

```bash
python src/evaluation.py
```

**Output:** Full metrics table with comparative analysis

---

## Feature Representations

### 1. Bag of Words (BoW)
- **Representation:** Word count vectors
- **Used by:** Multinomial Naive Bayes, Logistic Regression
- **Example:** "free free money" → [2, 0, 1, ...]

### 2. Bernoulli (Binary)
- **Representation:** Binary presence/absence vectors
- **Used by:** Bernoulli Naive Bayes, Logistic Regression
- **Example:** "free free money" → [1, 0, 1, ...]

---

## Algorithm Details

### Multinomial Naive Bayes
- **Complexity:** O(nd) training, O(nd) prediction
- **Smoothing:** Laplace (alpha=1.0)
- **Best for:** Word count features, text classification

### Bernoulli Naive Bayes
- **Complexity:** O(nd) training, O(nd) prediction
- **Smoothing:** Laplace (alpha=1.0)
- **Best for:** Binary features, document presence/absence

### Logistic Regression
- **Complexity:** O(Tnd) total (T iterations, n features, d samples)
- **Optimization:** Gradient ascent with vectorized operations
- **Regularization:** L2 (elastic net with lambda parameter)
- **Hyperparameters:** Learning rate, max iterations, lambda
- **Note:** Bias term (w₀) is NOT regularized

---

## Results Summary

Based on the final output:

| Algorithm                           | Average Accuracy |
|-------------------------------------|------------------|
| **Logistic Regression (Bernoulli)** | **96.03%**       |
| Logistic Regression (BoW)           | 95.63%           |
| Multinomial Naive Bayes             | 95.14%           |
| Bernoulli Naive Bayes               | 88.62%           |

### Key Findings:
- Logistic Regression with Bernoulli features achieved the best overall performance
- After hyperparameter tuning, LR outperformed Naive Bayes
- Best hyperparameters: learning_rate=0.1, lambda=0.0 (minimal/no regularization)
- All models achieve >88% accuracy, demonstrating robust spam detection

### Per-Dataset Best Performers:
- **enron1:** Logistic Regression (Bernoulli) - 95.39%
- **enron2:** Logistic Regression (Bernoulli) - 95.82%
- **enron4:** Multinomial Naive Bayes - 97.42%

---

## Implementation Notes

### Complexity Requirements
- All algorithms achieve **O(nd) complexity per iteration**
- Vectorized matrix operations (NumPy) instead of explicit loops
- Example: `X.T @ error` instead of nested for loops

### Libraries Used
- **NumPy:** Array operations and linear algebra
- **Pandas:** CSV reading/writing
- **NLTK:** Stopwords for text preprocessing
- **No ML libraries:** No scikit-learn, TensorFlow, or similar for core algorithms

### Key Design Decisions
1. **Vocabulary built only on training data** to prevent data leakage
2. **Log probabilities** used throughout to prevent numerical underflow
3. **Laplace smoothing** (alpha=1.0) for handling unseen words
4. **Clipping in sigmoid** to prevent overflow in extreme values
5. **Hyperparameter tuning** via grid search for optimal Logistic Regression performance

---

## Troubleshooting

### "Results directory not found"
Run `python src/data_preprocessing.py` first to generate feature matrices.

### "ModuleNotFoundError: No module named 'nltk'"
Install dependencies: `pip install nltk pandas numpy`

### "Resource stopwords not found"
Download NLTK data: `python -c "import nltk; nltk.download('stopwords')"`

### "FileNotFoundError: Could not find training data"
Ensure your data directory structure matches:
```
data/enron1_train/train/spam/
data/enron1_train/train/ham/
data/enron1_test/test/spam/
data/enron1_test/test/ham/
data/enron2_train/train/spam/
data/enron2_train/train/ham/
data/enron2_test/test/spam/
data/enron2_test/test/ham/
data/enron4_train/train/spam/
data/enron4_train/train/ham/
data/enron4_test/test/spam/
data/enron4_test/test/ham/
```
The code expects the nested `/train` and `/test` subdirectories as shown above.

### Low accuracy results
Check that:
- All 12 CSV files exist in `results/` directory
- Hyperparameters are tuned (especially for Logistic Regression)
- Data preprocessing completed successfully

---

## Citation

This project uses:
- **NLTK (nltk>=3.7)** for stopwords: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.
- **NumPy (numpy>=1.21.0)** for numerical operations: Harris, C.R., Millman, K.J., et al. (2020). Array programming with NumPy. Nature, 585, 357–362.
- **Pandas (pandas>=1.3.0)** for data manipulation: McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56.

