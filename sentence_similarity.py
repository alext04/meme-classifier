from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model= SentenceTransformer('bert-base-nli-mean-tokens')

vector=model.encode(["a man standing over a city","muslim invasion"])

print(cosine_similarity([vector[0]],vector[1:]))