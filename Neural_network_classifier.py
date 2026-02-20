import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)


class ShallowNeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1):
        self.lr = lr
        self.layers = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            activations.append(a)
        return activations

    def train(self, X, Y, epochs=50, batch_size=32):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X, Y = X[idx], Y[idx]

            for start in range(0, len(X), batch_size):
                end = start + batch_size
                xb = X[start:end]
                yb = Y[start:end]

                activations = self.forward(xb)
                output = activations[-1]

                error = yb - output
                delta = error * sigmoid_derivative(output)

                deltas = [delta]
                for i in reversed(range(len(self.weights) - 1)):
                    delta = np.dot(deltas[0], self.weights[i + 1].T) * sigmoid_derivative(activations[i + 1])
                    deltas.insert(0, delta)

                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * np.dot(activations[i].T, deltas[i])
                    self.biases[i] += self.lr * np.sum(deltas[i], axis=0, keepdims=True)

    def predict(self, X):
        output = self.forward(X)[-1]
        return np.argmax(output, axis=1)

def generate_class(modes, samples, label):
    X, y = [], []
    for _ in range(modes):
        mean = np.random.uniform(-1, 1, size=2)
        cov = np.diag(np.random.uniform(0.05, 0.2, size=2))
        pts = np.random.multivariate_normal(mean, cov, samples)
        X.append(pts)
        y.extend([label] * samples)
    return np.vstack(X), np.array(y)

def generate_data(m0, m1, samples):
    X0, y0 = generate_class(m0, samples, 0)
    X1, y1 = generate_class(m1, samples, 1)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    Y = np.zeros((len(y), 2))
    Y[np.arange(len(y)), y] = 1
    return X, Y, y

def plot_boundary(net, X, labels):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = net.predict(grid)
    Z = preds.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[labels == 0][:, 0], X[labels == 0][:, 1], c="blue", label="Class 0")
    ax.scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], c="red", label="Class 1")
    ax.set_title("Decision Boundary – Shallow Neural Network")
    ax.legend()
    return fig


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_idx_images(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28*28) / 255.0

def load_idx_labels(path):
    with open(path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def one_hot(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes))
    Y[np.arange(len(labels)), labels] = 1
    return Y

def load_mnist(limit=5000):
    X_path = os.path.join(BASE_DIR, "train-images.idx3-ubyte") 
    y_path = os.path.join(BASE_DIR, "train-labels.idx1-ubyte")
    X = load_idx_images(X_path)
    y = load_idx_labels(y_path)
    X = X[:limit]
    y = y[:limit]
    Y = one_hot(y, 10)
    return X, Y, y


st.title("Shallow Neural Network")

mode = st.selectbox("Select Mode", ["Gaussian 2D Dataset", "MNIST Dataset"])

lr = st.slider("Learning Rate η", 0.001, 1.0, 0.1)
epochs = st.slider("Epochs", 10, 200, 50)
batch_size = st.slider("Batch Size", 8, 128, 32)

if mode == "Gaussian 2D Dataset":
    st.subheader("Gaussian Dataset Classification")
    m0 = st.number_input("Modes Class 0", 1, 5, 2)
    m1 = st.number_input("Modes Class 1", 1, 5, 2)
    samples = st.number_input("Samples per Mode", 20, 300, 100)
    hidden_layers = st.slider("Hidden Layers (1–3)", 1, 3, 1)
    neurons = st.slider("Neurons per Hidden Layer", 2, 20, 8)

    if st.button("Generate Data & Train Network"):
        X, Y, labels = generate_data(m0, m1, samples)
        layer_sizes = [2] + [neurons] * hidden_layers + [2]
        net = ShallowNeuralNetwork(layer_sizes, lr=lr)
        net.train(X, Y, epochs=epochs, batch_size=batch_size)
        fig = plot_boundary(net, X, labels)
        st.pyplot(fig)

if mode == "MNIST Dataset":
    st.subheader("MNIST Digit Classification (Flattened Vectors)")
    limit = st.slider("Training Samples Used", 1000, 20000, 5000)
    hidden_layers = st.slider("Hidden Layers (1–4)", 1, 4, 2)
    neurons = st.slider("Neurons per Hidden Layer", 16, 128, 64)

    if st.button("Train Network on MNIST"):
        X, Y, labels = load_mnist(limit)
        layer_sizes = [784] + [neurons] * hidden_layers + [10]
        net = ShallowNeuralNetwork(layer_sizes, lr=lr)
        net.train(X, Y, epochs=epochs, batch_size=batch_size)
        preds = net.predict(X)
        acc = np.mean(preds == labels)
        st.success(f"Training Accuracy on MNIST subset: {acc*100:.2f}%")
