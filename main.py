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

contentsEmbeddings = model.encode(myExiDf['exhibition_contents'].tolist())
print(len(contentsEmbeddings))
def rcm(user_input, topN=5):
    matchingRows = myExiDf[myExiDf['exhibition_title'] == user_input]

    if matchingRows.empty:
        print(f"No exhibition found with title '{user_input}'")
        return [], []

    inputContentIndex = matchingRows.index[0]
    inputContentEmbedding = contentsEmbeddings[inputContentIndex]
    similarities = util.cos_sim(inputContentEmbedding, contentsEmbeddings).flatten()
    print(inputContentIndex)
    print(inputContentEmbedding)
    print("Length of similarities:", len(similarities))
    print("Similarities array:", similarities)

    if len(similarities) < topN:
        print("Not enough exhibitions for comparison.")
        return [], []

    similarIndices = (-similarities).argsort()[:min(topN, len(similarities))]
    similarIndices = similarIndices.tolist()
    similarTitles = myExiDf.loc[similarIndices, 'exhibition_title'].tolist()
    similaritiesList = similarities[similarIndices].tolist()
    return similarTitles, similaritiesList


user_input = input("Enter an exhibition title: ")

topTitles, similarities = rcm(user_input)
print(f"Top 5 similar titles for '{user_input}' based on content similarity:")
for title, similarity in zip(topTitles, similarities):
    print(f"{title} - Similarity: {similarity:.4f}")
