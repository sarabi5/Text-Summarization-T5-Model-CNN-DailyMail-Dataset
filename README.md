# Text-Summarization-T5-Model-CNN-DailyMail-Dataset
This project implements an end-to-end pipeline for abstractive text summarization using the T5-small model from Hugging Face's Transformers library. The model is fine-tuned on a sampled subset of the CNN/DailyMail dataset for efficient training within a Google Colab environment.

https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Key Features:

Reads and preprocesses train, validation, and test CSV files uploaded to Google Drive.
Applies T5 tokenizer with truncation and padding for model compatibility.
Converts pandas DataFrames into Hugging Face Dataset format with PyTorch tensors.
Fine-tunes T5ForConditionalGeneration using the Hugging Face Trainer API.
Computes ROUGE scores to evaluate summarization quality.
Includes a summarize() function to test the model on custom inputs.
Saves processed datasets and trained model/tokenizer to disk for future use.
Libraries Used: Hugging Face Ecosystem (including transformers, datasets, accelerate, and evaluate), rouge-score, sentencepiece, pandas, and numpy

Please see the outputs of the code in the following link in .inpy format: https://drive.google.com/file/d/1ImyqU3s9FTe9ZGN7QtYCIG-gIz3Lxr4l/view?usp=drive_link
