import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import torch
import torch.nn as nn
import nltk
import spacy

# Numpy: create random data
np_data = np.random.rand(100, 4)

# Pandas: create DataFrame
df = pd.DataFrame(np_data, columns=['A', 'B', 'C', 'D'])

# Matplotlib & Seaborn: plot data
plt.figure(figsize=(8, 4))
sns.histplot(df['A'], kde=True)
plt.title('Distribution of column A')
plt.show()

# Sklearn: load iris dataset and train a classifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Sklearn accuracy:", clf.score(X_test, y_test))

# TensorFlow: simple dense model
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=3, verbose=0)
tf_loss, tf_acc = tf_model.evaluate(X_test, y_test, verbose=0)
print("TensorFlow accuracy:", tf_acc)

# PyTorch: simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

torch_model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
for epoch in range(3):
    optimizer.zero_grad()
    outputs = torch_model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()
print("PyTorch training loss:", loss.item())

# NLTK: tokenize a sentence
nltk.download('punkt', quiet=True)
sentence = "Natural Language Processing with NLTK and spaCy is fun!"
tokens = nltk.word_tokenize(sentence)
print("NLTK tokens:", tokens)

# spaCy: named entity recognition
nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)
print("spaCy entities:", [(ent.text, ent.label_) for ent in doc.ents])
