import tensorflow as tf
from tensorboard.plugins import projector

def create_embedding_visualization(model, word_index, log_dir):
weights = model.get_layer(index=0).get_weights()[0]
with open(log_dir + '/metadata.tsv', 'w') as metadata_file:
for word, index in word_index.items():
metadata_file.write('{}\n'.format(word))
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'word_embeddings'
embedding.metadata_path = 'metadata.tsv'
checkpoint = tf.train.Checkpoint(embedding=embedding)
checkpoint.save(log_dir + '/embedding.ckpt')
projector.visualize_embeddings(log_dir, config)