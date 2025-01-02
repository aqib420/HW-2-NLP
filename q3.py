import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Loading the training and test datasets
train_file_path = 'sentiment_train_dataset.csv'
test_file_path = 'sentiment_test_dataset.csv'

# Reading datasets
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Preprocessing function: text cleaning, tokenization, and vocabulary building
def preprocess_text(text):
    # Convert to lowercase and remove any special characters or punctuations
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens

# Preprocess the training and test sentences
train_data['tokens'] = train_data['sentence'].apply(preprocess_text)
test_data['tokens'] = test_data['sentence'].apply(preprocess_text)

# Vocabulary building
vocab = set([word for tokens in train_data['tokens'] for word in tokens])
vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}

# Convert tokenized text to one-hot encoded vectors
def tokens_to_one_hot(tokens, vocab_size, word_to_index):
    vector = np.zeros(vocab_size)
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] = 1
    return vector

# Prepare the training and testing feature matrices
X_train = np.array([tokens_to_one_hot(tokens, vocab_size, word_to_index) for tokens in train_data['tokens']])
y_train = train_data['label'].values

X_test = np.array([tokens_to_one_hot(tokens, vocab_size, word_to_index) for tokens in test_data['tokens']])
y_test = test_data['label'].values

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Neural network class
class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    # Backward propagation and parameter update
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # Calculate output layer error
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Calculate hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # Train the model
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            A2 = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
                print(f'Epoch {epoch}, Loss: {loss}')

    # Predict on new data
    def predict(self, X):
        A2 = self.forward(X)
        return np.where(A2 > 0.5, 1, 0)

# Model initialization
input_size = vocab_size
hidden_size = 128
output_size = 1
epochs = 1000
learning_rate = 0.01

# Create the ANN model
ann = ANN(input_size, hidden_size, output_size)

# Train the model
ann.train(X_train, y_train, epochs, learning_rate)

# Evaluate the model on the test data
y_pred = ann.predict(X_test)

# Evaluation: accuracy, confusion matrix, precision, recall, f1-score
accuracy = np.mean(y_pred.flatten() == y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output the evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
