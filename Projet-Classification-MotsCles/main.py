import os
from dataclasses import dataclass
from models.word_embedding import create_word_embeddings
from models.semantic_clustering import cluster_keywords, train_model
from models.visualization import create_embedding_visualization

@dataclass
class Config:
data_file: str = 'data/mots-cles.xlsx'
stopword_file: str = 'data/stopwords.txt'
log_dir: str = 'output/tensorboard_logs'

def main(config):
word_index, word_embeddings, data = create_word_embeddings(config.data_file)
labels = cluster_keywords