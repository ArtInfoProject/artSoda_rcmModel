import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

file_path = 'exhibition_board.json'

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for exhibition in data:
    exhibition_contents = exhibition['exhibition_contents']
    cleaned_sentence = re.sub("[^ 0-9가-힣A-Za-z]", "", exhibition_contents)
    cleaned_sentence = re.sub('\s+', ' ', cleaned_sentence).strip()
    exhibition['exhibition_contents'] = cleaned_sentence

myExiDf = pd.DataFrame(data, columns=['exhibition_title', 'exhibition_contents'])
# print(myExiDf['exhibition_contents'])

content_embeddings = model.encode(myExiDf['exhibition_contents'].tolist())


def find_similar_titles(user_input, top_n=5):
    matching_rows = myExiDf[myExiDf['exhibition_title'] == user_input]

    if matching_rows.empty:
        print(f"No exhibition found with title '{user_input}'")
        return [], []

    input_content_index = matching_rows.index[0]
    input_content_embedding = content_embeddings[input_content_index]
    similarities = util.cos_sim(input_content_embedding, content_embeddings).flatten()

    print("Length of similarities:", len(similarities))
    print("Similarities array:", similarities)

    if len(similarities) < top_n:
        print("Not enough exhibitions for comparison.")
        return [], []

    similar_indices = (-similarities).argsort()[:min(top_n, len(similarities))]
    similar_indices = similar_indices.tolist()
    similar_titles = myExiDf.loc[similar_indices, 'exhibition_title'].tolist()
    similar_similarities = similarities[similar_indices].tolist()
    return similar_titles, similar_similarities


user_input = input("Enter an exhibition title: ")

# Find and print similar titles with similarities
top_similar_titles, similarities = find_similar_titles(user_input)
print(f"Top 5 similar titles for '{user_input}' based on content similarity:")
for title, similarity in zip(top_similar_titles, similarities):
    print(f"{title} - Similarity: {similarity:.4f}")
