import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from seqeval.scheme import BILOU
from seqeval.metrics import classification_report, f1_score
import gradio as gr
from collections import Counter

# Configure TensorFlow for M1 MPS
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("MPS GPU available and configured.")
else:
    print("No GPU detected, using CPU.")

# Data Loading
def load_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label_seq = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label_seq)
                    sentence = []
                    label_seq = []
                continue
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            if "-" not in line:
                continue
            token, tag = line.rsplit("-", 1)
            sentence.append(token)
            label_seq.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels

# Load data
file_path = "dataset_betawi_split.txt"
sentences, labels = load_data(file_path)

# Feature Engineering
def build_vocab(sentences, labels):
    word_counts = Counter(word for sent in sentences for word in sent)
    tag_counts = Counter(tag for seq in labels for tag in seq)
    word2idx = {word: idx + 2 for idx, word in enumerate(word_counts)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_counts.keys()))}
    return word2idx, tag2idx

word2idx, tag2idx = build_vocab(sentences, labels)
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Load pre-trained model
model = load_model("best_bilstm_model.h5")
model.summary()

# Prediction
def preprocess_text(text):
    words = text.split()
    x = [[word2idx.get(word, word2idx["<UNK>"]) for word in words]]
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=128, padding='post', dtype='int32')  # Integer indices
    
    # Tambahkan dimensi embedding (50) dengan nilai dummy (akan diganti dengan embedding asli)
    x = tf.cast(x, tf.float32)  # Konversi ke float32
    x = tf.expand_dims(x, axis=-1)  # Tambah dimensi sementara
    x = tf.tile(x, [1, 1, 50])  # Ulangi untuk 50 dimensi dengan nilai sementara
    return x

def predict_ner(text):
    x = preprocess_text(text)
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = [[idx2tag[idx] for idx in seq if idx != 0] for seq in y_pred]
    
    # Agregasi label
    labels = y_pred[0]  # Ambil label untuk kalimat pertama
    label_counts = {}
    for label in labels:
        category = label.split('_')[0]  # Ambil kategori (e.g., FUEL, SERVICE)
        if category in ['FUEL', 'MACHINE', 'OTHERS', 'PART', 'PRICE', 'SERVICE']:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    # Prioritaskan berdasarkan kategori yang ada
    priority_categories = ['FUEL', 'MACHINE', 'OTHERS', 'PART', 'PRICE', 'SERVICE']
    for category in priority_categories:
        for polar in ['POSITIVE', 'NEGATIVE']:
            if f"{category}_{polar}" in label_counts:
                return f"Predicted label for sentence: {category}_{polar}"
    
    return "No dominant label found"

# Training Visualization
def plot_training_history():
    history_df = pd.read_csv("history_BiLSTM.csv")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Val Loss')
    plt.title('BiLSTM - Training History (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Val Accuracy')
    plt.title('BiLSTM - Training History (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    return "Training history plot saved as 'training_plot.png'"

# Gradio Interface
with gr.Blocks(title="NLP Model Deployment") as demo:
    gr.Markdown("# NLP Model Deployment for UAS")
    gr.Markdown("Demonstrates data loading, feature engineering, training visualization, and prediction using a pre-trained BiLSTM model.")

    with gr.Tab("Data Loading"):
        gr.Markdown("Preview of the loaded dataset.")
        data_output = gr.Textbox(label="Sample Data", value=str(sentences[:2]), interactive=False)

    with gr.Tab("Feature Engineering"):
        gr.Markdown("Preprocessed vocabulary and tag mapping.")
        vocab_output = gr.Textbox(label="Vocabulary Size", value=len(word2idx), interactive=False)
        tag_output = gr.Textbox(label="Tag Size", value=len(tag2idx), interactive=False)

    with gr.Tab("Training Visualization"):
        gr.Markdown("Visualize the pre-trained model’s training history.")
        plot_button = gr.Button("Generate Plot")
        plot_output = gr.Textbox(label="Plot Status")
        plot_button.click(fn=plot_training_history, inputs=None, outputs=plot_output)

    with gr.Tab("Prediction"):
        gr.Markdown("Predict NER tags for new text input.")
        input_text = gr.Textbox(label="Input Text", value="Ane pake Honda Jazz")
        predict_button = gr.Button("Predict")
        output_text = gr.JSON(label="Predicted Tags")
        predict_button.click(fn=predict_ner, inputs=input_text, outputs=output_text)

demo.launch()