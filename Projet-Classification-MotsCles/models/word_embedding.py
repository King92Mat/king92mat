import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_word_embeddings(data_file):
data = pd.read_excel(data_file)
stop_words = set(stopwords.words('french'))
tokenized_data = [nltk.word_tokenize(str(keyword)) for keyword in data['mot-clé']]
filtered_data = []
for tokens in tokenized_data:
filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
filtered_data.append(' '.join(filtered_tokens))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_data)
sequences = tokenizer.texts_to_sequences(filtered_data)
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences)
return word_index, padded_sequences, data