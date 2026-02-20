🧠 Shallow Neural Network – 2D & MNIST Classifier

📌 Overview

This project is an interactive Streamlit demo of a shallow neural network for classification. 

It supports two modes:

Gaussian 2D Dataset – Generate synthetic 2D data with configurable modes and visualize the decision boundary.

MNIST Dataset – Classify handwritten digits using a shallow fully connected neural network.

Users can configure learning rate, epochs, batch size, hidden layers, and neurons, and observe model performance visually.

🧠 Features

Train a shallow neural network on 2D Gaussian datasets

Train on MNIST handwritten digits

Adjustable hidden layers and neurons

Configurable learning rate, epochs, and batch size

Decision boundary visualization (2D mode)

Training accuracy display (MNIST mode)

📂 MNIST Dataset Setup (Required for MNIST Mode)

If you want to use the MNIST Dataset mode, you must manually download the dataset files.

Download the following dataset zip file from the official MNIST website:

👉 (https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

Required files:

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

Steps:

Download all dataset zip file

Extract it (unzip)

Place the extracted .idx files in the same folder as:

Neural_network_classifier.py

The app will automatically load the training files when MNIST mode is selected.

🚀 How to Run

1️⃣ Clone the repository

git clone https://github.com/saur4bbh/Shallow-Neural-Network---2D-MNIST-Classifier

2️⃣ Navigate into the folder

cd Shallow-Neural-Network---2D-MNIST-Classifier

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the Streamlit app

streamlit run Neural_network_classifier.py

🔮 Future Improvements

Add loss visualization graph

Implement model saving/loading

Extend to deeper neural networks
