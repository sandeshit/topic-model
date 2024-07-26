from bertopic import BERTopic
from dataset import docs
from embedding import sentence_model
from dimension import umap_model
from cluster import hdbscan_model


topic_model = BERTopic(embedding_model= sentence_model, umap_model= umap_model, hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(docs[0:1000])

topic_model.get_topic_info()