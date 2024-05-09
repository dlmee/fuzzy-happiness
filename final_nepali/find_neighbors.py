import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_vectors(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        vectors = json.load(f)
    return vectors

def get_closest_candidates(query_vector, vectors, k=5):
    similarities = {}
    for stem, vector in vectors.items():
        similarity = cosine_similarity([query_vector], [vector])[0][0]
        similarities[stem] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:k]

if __name__ == "__main__":
    vectors_file_path = "final/stem_vectors.json"  # Replace with the path to your stem vectors file
    k = 6  # Specify the number of closest candidates to retrieve

    vectors = load_vectors(vectors_file_path)
    random_stem = random.choice(list(vectors.keys()))
    query_vector = np.array(vectors[random_stem])

    closest_candidates = get_closest_candidates(query_vector, vectors, k)

    print(f"Random Stem: {random_stem}")
    print(f"Top {k} closest candidates:")
    for candidate, similarity in closest_candidates:
        print(f"{candidate}: {similarity}")
