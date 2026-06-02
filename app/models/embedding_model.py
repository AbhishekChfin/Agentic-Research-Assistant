from sentence_transformers import SentenceTransformer

model_name = 'BAAI/bge-base-en-v1.5'
def get_embedding_model():
    return SentenceTransformer(
        model_name
    )