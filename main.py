import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

file_path = 'exhibition_board.json'

with open(file_path, 'r',encoding='utf-8') as file:
    data = json.load(file)

def clean_sentence(sentence):
    cleaned_sentence = re.sub("[^ 0-9가-힣A-Za-z]", "", sentence)
    cleaned_sentence = re.sub('\s+', ' ', cleaned_sentence).strip()
    return cleaned_sentence

for exhibition in data:
    exhibition_contents = exhibition['exhibition_contents']
    cleaned_sentence = re.sub("[^ 0-9가-힣A-Za-z]", "", exhibition_contents)
    cleaned_sentence = re.sub('\s+', ' ', cleaned_sentence).strip()
    # cleaned_contents = exhibition_contents.replace('\r', '')
    # cleaned_contents = cleaned_contents.replace('\r\r', ' ')
    # cleaned_contents = cleaned_contents.replace('●', '')
    exhibition['exhibition_contents'] = cleaned_sentence

myExiDf = pd.DataFrame(data, columns=['exhibition_title', 'exhibition_contents'])
print(myExiDf['exhibition_contents'])



