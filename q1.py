import numpy as np
from collections import defaultdict

class NGram:
    def __init__(self, corpus) -> None:
        self.corpus = corpus
        self.tokens = []
        self.vocab = set()  # to store unique words
        self.ngrams_counts = defaultdict(int)  # to store counts of ngrams
        self.unigrams_counts = defaultdict(int)  # to store counts of unigrams

    def preprocess(self) -> list:
        # cleaning special characters
        special_characters = ['.', '~', '!', '@', '#', '$', '%', '^', '&', '*',
                              '(', ')', '-', '_', '+', '=', '`', '|', '\\', ']', '}', '[', '{', '"', ':', ';', '/', '?', '>', ',', '<']
        
        # replace special characters with space
        for char in special_characters:
            self.corpus = self.corpus.replace(char, ' ')
        
        # lowercase and tokenize
        self.tokens = self.corpus.lower().split()
        self.vocab = set(self.tokens)  # update vocabulary
        return self.tokens
    
    def generate_ngrams(self, n) -> list:
        ngrams = []
        # Generate n-grams
        for i in range(len(self.tokens) - n + 1):
            ngram = tuple(self.tokens[i:i+n])
            ngrams.append(ngram)
            # Update n-grams counts
            self.ngrams_counts[ngram] += 1
            self.unigrams_counts[ngram[:-1]] += 1  # Increment counts for the (n-1)-gram
        return ngrams

    def calculate_probabilities(self, n) -> dict:
        probabilities = defaultdict(dict)
        vocab_size = len(self.vocab)

        # Calculate probabilities with add-one smoothing
        for ngram, count in self.ngrams_counts.items():
            prefix = ngram[:-1]  # (n-1)-gram
            probabilities[prefix][ngram[-1]] = (count + 1) / (self.unigrams_counts[prefix] + vocab_size)

        return probabilities
    
    def perplexity(self, test_tokens, n, probabilities) -> float:
        log_prob_sum = 0
        N = len(test_tokens)
        
        # Calculate log probabilities for the test tokens
        for i in range(n-1, N):
            prefix = tuple(test_tokens[i-n+1:i])  # Get the (n-1)-gram prefix
            word = test_tokens[i]
            prob = probabilities.get(prefix, {}).get(word, 1 / (self.unigrams_counts[prefix] + len(self.vocab)))  # Default probability for unseen words
            log_prob_sum += np.log(prob)

        # Calculate perplexity
        perplexity = np.exp(-log_prob_sum / (N - n + 1))
        return perplexity

# Boilerplate code
if __name__ == "__main__":
    # Load training data
    with open("train.txt", "r") as file:
        train_corpus = file.read()
    
    # Load test data
    with open("test.txt", "r") as file:
        test_corpus = file.read()
    
    # Preprocess the training data
    ngram_model = NGram(train_corpus)
    train_tokens = ngram_model.preprocess()

    # Preprocess the test data
    test_ngram_model = NGram(test_corpus)
    test_tokens = test_ngram_model.preprocess()

    # Test different N-Gram models (Unigram, Bigram, Trigram)
    for n in [1, 2, 3]:  # Unigram, Bigram, Trigram
        ngram_model = NGram(train_corpus)  # Re-initialize for each n
        train_tokens = ngram_model.preprocess()  # Preprocess training tokens
        ngram_model.generate_ngrams(n)  # Generate n-grams
        probabilities = ngram_model.calculate_probabilities(n)  # Calculate probabilities
        perplexity = ngram_model.perplexity(test_tokens, n, probabilities)  # Calculate perplexity
        print(f"Perplexity for n={n}: {perplexity:.4f}")  # Print with 4 decimal places
