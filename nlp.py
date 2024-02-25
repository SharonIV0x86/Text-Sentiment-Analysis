import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

reviews = [
    "This movie was fantastic!",
    "Great movie, I really enjoyed it.",
    "Terrible acting, boring plot.",
    "Worst movie I've ever seen."
] #create more of such and feed them to model ez.

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

def train_pytorch_model(X_train, y_train, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    return accuracy, precision, recall, f1

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 32
hidden_dim = 16
output_dim = 1

model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_torch = torch.LongTensor(X_train)
y_train_torch = torch.FloatTensor(y_train.reshape(-1, 1))

train_pytorch_model(X_train_torch, y_train_torch, model, criterion, optimizer, epochs=10)

keras_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=10),
    LSTM(hidden_dim),
    Dense(1, activation='sigmoid')
])
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

keras_model.fit(X_train, y_train, epochs=10, verbose=1)

keras_pred = (keras_model.predict(X_test) > 0.5).astype(int)
print("Evaluation on Keras model:")
evaluate_model(y_test, keras_pred)

tf_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=10),
    LSTM(hidden_dim),
    Dense(1, activation='sigmoid')
])
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, verbose=1)

tf_pred = (tf_model.predict(X_test) > 0.5).astype(int)
print("Evaluation on TensorFlow model:")
evaluate_model(y_test, tf_pred)
