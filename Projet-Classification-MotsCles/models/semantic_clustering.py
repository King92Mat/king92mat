from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.cluster import KMeans

def cluster_keywords(word_embeddings):
kmeans = KMeans(n_clusters=5)
kmeans.fit(word_embeddings)
return kmeans.labels_

def build_model(input_dim):
model = Sequential()
model.add(Dense(256, input_dim=input_dim, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
return model

def train_model(word_embeddings, labels):
model = build_model(word_embeddings.shape[1])
model.fit(word_embeddings, labels, epochs=50, batch_size=16)
return model