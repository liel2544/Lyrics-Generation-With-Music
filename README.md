# Lyrics Generation with LSTM and MIDI Integration

<img src="Lyrics Generation_visual.webp" alt="Lyrics Generation Project Overview" width="600"/>

## Overview
This project explores automatic song lyrics generation using Recurrent Neural Networks (LSTM) integrated with melody data from MIDI files. The model predicts the next word in a lyrics sequence while aligning with the song's melody, combining Word2Vec embeddings and MIDI features.

## Key Features
- LSTM-based deep learning models.
- Integration of textual lyrics with MIDI musical features.
- Two model variants using different MIDI representations.
- Word2Vec pre-trained embeddings.
- Top-K sampling strategy for diverse lyric generation.
- TensorBoard visualization of training/validation performance.

## Technologies Used
- Python
- PyTorch
- Word2Vec (Google News embeddings)
- pretty_midi
- TensorBoard
- Pandas, NumPy

## Project Structure
- `main_notebook.ipynb`: Main project pipeline and experiments.
- `lyrics_analisys.ipynb`: Additional evaluation and analysis.
- `LSTM.py`: LSTM model architecture.
- `train_val_test.py`: Model training, validation, and testing logic.
- `data_sets_loaders.py`: Custom PyTorch Dataset & DataLoader.
- `preprocess_csv.py`: Lyrics preprocessing and embedding.
- `preprocess_midi.py`: MIDI feature extraction and integration.
- `real_dataset.csv`: Processed lyrics and melody dataset.
- `Report.pdf`: Full project report, methodology, and results.

## Dataset Notice
All necessary project files and datasets are included in this repository **except for the training `.pkl` files**, which were excluded due to file size limitations.

To recreate the training dataset locally, please run:
```bash
python preprocess_csv.py
python preprocess_midi.py
```
Ensure you provide the correct paths to the original lyrics CSV and MIDI files.

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/your-username/lyrics-generation-lstm-midi.git
cd lyrics-generation-lstm-midi
```
2. Install dependencies:
```bash
pip install torch pandas numpy gensim pretty_midi matplotlib
```
3. Run the notebooks:
```bash
jupyter notebook main_notebook.ipynb
```

## Highlights
- Sequential data processing using PyTorch Datasets.
- Two model variants:
  - Model 1: Word2Vec + basic MIDI features (beats, instruments).
  - Model 2: Word2Vec + extended MIDI features (piano roll).
- Evaluation based on loss trends and generated lyric quality.

---
This project bridges music and AI through deep learning, combining melody and language to explore creative text generation.

