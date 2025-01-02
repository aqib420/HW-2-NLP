import os
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def load_data(pos_dir, neg_dir, num_samples):
    """Load the specified number of positive and negative samples."""
    pos_reviews = []
    neg_reviews = []

    # Load positive reviews
    for filename in os.listdir(pos_dir)[:num_samples]:
        with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as file:
            pos_reviews.append(file.read())
    
    # Load negative reviews
    for filename in os.listdir(neg_dir)[:num_samples]:
        with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as file:
            neg_reviews.append(file.read())

    return pos_reviews, neg_reviews

def preprocess_text(text):
    """Clean and tokenize the text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenization
    return tokens

def build_vocabulary(reviews):
    """Build a vocabulary of unique words from the reviews."""
    vocabulary = set()
    for review in reviews:
        tokens = preprocess_text(review)
        vocabulary.update(tokens)
    return vocabulary

def feature_extraction(reviews, vocabulary):
    """Extract features (word frequencies) from the reviews."""
    features = []
    for review in reviews:
        tokens = preprocess_text(review)
        word_count = defaultdict(int)
        for token in tokens:
            if token in vocabulary:
                word_count[token] += 1
        features.append(word_count)
    return features

class NaiveBayes:
    def __init__(self):
        self.prior_prob = {}
        self.likelihood = defaultdict(lambda: defaultdict(int))
        self.vocabulary_size = 0

    def train(self, pos_reviews, neg_reviews):
        """Train the Naive Bayes classifier."""
        total_reviews = len(pos_reviews) + len(neg_reviews)
        self.prior_prob['pos'] = len(pos_reviews) / total_reviews
        self.prior_prob['neg'] = len(neg_reviews) / total_reviews
        
        pos_word_count = sum(len(preprocess_text(review)) for review in pos_reviews)
        neg_word_count = sum(len(preprocess_text(review)) for review in neg_reviews)

        self.vocabulary_size = len(set(build_vocabulary(pos_reviews).union(build_vocabulary(neg_reviews))))

        # Calculate likelihoods
        for review in pos_reviews:
            tokens = preprocess_text(review)
            for token in tokens:
                self.likelihood['pos'][token] += 1
        
        for review in neg_reviews:
            tokens = preprocess_text(review)
            for token in tokens:
                self.likelihood['neg'][token] += 1

        # Apply Laplace smoothing
        for word in self.likelihood['pos']:
            self.likelihood['pos'][word] = (self.likelihood['pos'][word] + 1) / (pos_word_count + self.vocabulary_size)

        for word in self.likelihood['neg']:
            self.likelihood['neg'][word] = (self.likelihood['neg'][word] + 1) / (neg_word_count + self.vocabulary_size)

    def predict(self, review):
        """Predict the sentiment of a review."""
        tokens = preprocess_text(review)
        pos_score = np.log(self.prior_prob['pos'])
        neg_score = np.log(self.prior_prob['neg'])

        for token in tokens:
            pos_likelihood = self.likelihood['pos'].get(token, 1 / (self.vocabulary_size + 1))
            neg_likelihood = self.likelihood['neg'].get(token, 1 / (self.vocabulary_size + 1))
            pos_score += np.log(pos_likelihood)
            neg_score += np.log(neg_likelihood)

        return 'pos' if pos_score > neg_score else 'neg'

def evaluate_model(classifier, test_pos, test_neg):
    """Evaluate the model's performance."""
    test_reviews = test_pos + test_neg
    true_labels = ['pos'] * len(test_pos) + ['neg'] * len(test_neg)
    predictions = [classifier.predict(review) for review in test_reviews]

    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions, labels=['pos', 'neg'])
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, labels=['pos', 'neg'], average=None)

    return accuracy, conf_matrix, precision, recall, f1


def main():
    # Directories for positive and negative reviews
    pos_train_dir = 'train/pos'
    neg_train_dir = 'train/neg'
    pos_test_dir = 'test/pos'
    neg_test_dir = 'test/neg'
    
    # Load data
    pos_train, neg_train = load_data(pos_train_dir, neg_train_dir, 500)
    pos_test, neg_test = load_data(pos_test_dir, neg_test_dir, 100)

    # Initialize and train the Naive Bayes classifier
    classifier = NaiveBayes()
    classifier.train(pos_train, neg_train)

    # Evaluate the model
    accuracy, conf_matrix, precision, recall, f1 = evaluate_model(classifier, pos_test, neg_test)

    # Print evaluation results
    print(f'Accuracy: {accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print(f'Precision (pos, neg): {precision[0]:.4f}, {precision[1]:.4f}')
    print(f'Recall (pos, neg): {recall[0]:.4f}, {recall[1]:.4f}')
    print(f'F1 Score (pos, neg): {f1[0]:.4f}, {f1[1]:.4f}')

if __name__ == '__main__':
    main()
