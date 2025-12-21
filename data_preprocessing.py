import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords

class EmailPreprocessor:
    def __init__(self, min_word_freq=2, use_nltk_stopwords=True):
        self.min_word_freq = min_word_freq
        self.vocabulary = None
        
        if use_nltk_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))
        else:
            #  custom stopwords list
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
                'the', 'to', 'was', 'will', 'with', 'would', 'you', 'your', 'we',
                'i', 'me', 'my', 'they', 'them', 'their', 'this', 'these', 'those',
                'but', 'or', 'if', 'while', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
            }
        
    def clean_text(self, text):
        text = text.lower()
        
        # Remove email header labels but keep content
        text = re.sub(r'^(subject|from|to|date|cc|bcc):\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text):
        tokens = re.findall(r'\b[a-z]+\b', text)
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 1]
        return tokens
    
    def build_vocabulary(self, email_texts):
        all_words = []
        
        for text in email_texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)
            all_words.extend(tokens)
        
        word_counts = Counter(all_words)
        vocab_words = sorted([word for word, count in word_counts.items()  if count >= self.min_word_freq])
        
        self.vocabulary = {word: idx for idx, word in enumerate(vocab_words)}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Total tokens processed: {len(all_words)}")
        print(f"Unique words before filtering: {len(word_counts)}")
        return self.vocabulary
    
    def get_vocabulary_list(self):
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built yet.")
        return sorted(self.vocabulary.keys())
    
    def text_to_bow_vector(self, text):
        """Convert text to bag-of-words feature vector (word counts)."""
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary() first.")
        
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        
        feature_vector = np.zeros(len(self.vocabulary), dtype=int)
        
        for token in tokens:
            if token in self.vocabulary:
                feature_vector[self.vocabulary[token]] += 1
        
        return feature_vector
    
    def text_to_bernoulli_vector(self, text):
        """Convert text to binary feature vector (word presence/absence)."""
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary() first.")
        
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        token_set = set(tokens)
        
        feature_vector = np.zeros(len(self.vocabulary), dtype=int)
        
        for token in token_set:
            if token in self.vocabulary:
                feature_vector[self.vocabulary[token]] = 1
        
        return feature_vector

class DatasetLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.preprocessor = EmailPreprocessor()
    
    def load_emails_from_directory(self, directory_path):
        emails = []
        labels = []
        
        spam_path = os.path.join(directory_path, 'spam')
        if os.path.exists(spam_path):
            for filename in os.listdir(spam_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(spam_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            emails.append(f.read())
                            labels.append(1)
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        ham_path = os.path.join(directory_path, 'ham')
        if os.path.exists(ham_path):
            for filename in os.listdir(ham_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(ham_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            emails.append(f.read())
                            labels.append(0)
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        return emails, labels
    
    def create_feature_matrices(self, dataset_name):
        possible_train_paths = [
            os.path.join(self.base_path, f"{dataset_name}_train", "train"), 
            os.path.join(self.base_path, f"{dataset_name}_train"),
            os.path.join(self.base_path, f"{dataset_name}", "train"),
            os.path.join(self.base_path, f"{dataset_name}/train")
        ]
        
        possible_test_paths = [
            os.path.join(self.base_path, f"{dataset_name}_test", "test"),   
            os.path.join(self.base_path, f"{dataset_name}_test"),
            os.path.join(self.base_path, f"{dataset_name}", "test"),
            os.path.join(self.base_path, f"{dataset_name}/test")
        ]
        
        train_path = None
        test_path = None
        
        for path in possible_train_paths:
            if os.path.exists(path):
                train_path = path
                break
        
        for path in possible_test_paths:
            if os.path.exists(path):
                test_path = path
                break
        
        if not train_path:
            raise FileNotFoundError(f"Could not find training data for {dataset_name}. Tried: {possible_train_paths}")
        if not test_path:
            raise FileNotFoundError(f"Could not find test data for {dataset_name}. Tried: {possible_test_paths}")
        
        print(f"Processing {dataset_name}...")
        
        train_emails, train_labels = self.load_emails_from_directory(train_path)
        print(f"Loaded {len(train_emails)} training emails")
        
        test_emails, test_labels = self.load_emails_from_directory(test_path)
        print(f"Loaded {len(test_emails)} test emails")
        
        self.preprocessor.build_vocabulary(train_emails)
        
        print("Creating Bag of Words matrices...")
        train_bow_matrix = []
        for email in train_emails:
            train_bow_matrix.append(self.preprocessor.text_to_bow_vector(email))
        train_bow_matrix = np.array(train_bow_matrix)
        
        test_bow_matrix = []
        for email in test_emails:
            test_bow_matrix.append(self.preprocessor.text_to_bow_vector(email))
        test_bow_matrix = np.array(test_bow_matrix)
        
        print("Creating Bernoulli matrices...")
        train_bernoulli_matrix = []
        for email in train_emails:
            train_bernoulli_matrix.append(self.preprocessor.text_to_bernoulli_vector(email))
        train_bernoulli_matrix = np.array(train_bernoulli_matrix)
        
        test_bernoulli_matrix = []
        for email in test_emails:
            test_bernoulli_matrix.append(self.preprocessor.text_to_bernoulli_vector(email))
        test_bernoulli_matrix = np.array(test_bernoulli_matrix)
        
        return {
            'train_bow': (train_bow_matrix, np.array(train_labels)),
            'test_bow': (test_bow_matrix, np.array(test_labels)),
            'train_bernoulli': (train_bernoulli_matrix, np.array(train_labels)),
            'test_bernoulli': (test_bernoulli_matrix, np.array(test_labels)),
            'vocabulary': self.preprocessor.vocabulary
        }
    
    def save_to_csv(self, data_dict, dataset_name, output_dir='results'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        vocabulary = data_dict['vocabulary']
        
        train_bow_data, train_labels = data_dict['train_bow']
        df = pd.DataFrame(train_bow_data, columns=vocabulary)
        df['label'] = train_labels
        df.to_csv(os.path.join(output_dir, f'{dataset_name}_bow_train.csv'), index=False)
        
        test_bow_data, test_labels = data_dict['test_bow']
        df = pd.DataFrame(test_bow_data, columns=vocabulary)
        df['label'] = test_labels
        df.to_csv(os.path.join(output_dir, f'{dataset_name}_bow_test.csv'), index=False)
        
        train_bernoulli_data, train_labels = data_dict['train_bernoulli']
        df = pd.DataFrame(train_bernoulli_data, columns=vocabulary)
        df['label'] = train_labels
        df.to_csv(os.path.join(output_dir, f'{dataset_name}_bernoulli_train.csv'), index=False)
        
        test_bernoulli_data, test_labels = data_dict['test_bernoulli']
        df = pd.DataFrame(test_bernoulli_data, columns=vocabulary)
        df['label'] = test_labels
        df.to_csv(os.path.join(output_dir, f'{dataset_name}_bernoulli_test.csv'), index=False)
        
        print(f"CSV files saved for {dataset_name}")

def main():
    data_path = "data"
    loader = DatasetLoader(data_path)
    
    datasets = ['enron1', 'enron2', 'enron4']
    
    for dataset in datasets:
        try:
            data_dict = loader.create_feature_matrices(dataset)
            loader.save_to_csv(data_dict, dataset)
            print(f"Successfully processed {dataset}\n")
        except Exception as e:
            print(f"Error processing {dataset}: {e}\n")

if __name__ == "__main__":
    main()