from sqlSetting import exh_df
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
def preprocess_data():
    for index, exhibition in exh_df.iterrows():
        exhibition_contents = exhibition['exhibition_contents']
        cleaned_contents = re.sub("[^ 0-9가-힣A-Za-z]", "", exhibition_contents)
        cleaned_contents = re.sub('\s+', ' ', cleaned_contents).strip()
        exh_df.loc[index, 'exhibition_contents'] = cleaned_contents
    myExiDf = pd.DataFrame(exh_df, columns=['exhibition_title', 'exhibition_contents', 'exhibition_img'])
    contentsEmbeddings = model.encode(myExiDf['exhibition_contents'].tolist())
    return myExiDf, contentsEmbeddings

def recommend(user_input, myExiDf, contentsEmbeddings, topN=6):
    matchingRows = myExiDf[myExiDf['exhibition_title'] == user_input]
    if matchingRows.empty:
        raise ValueError(f"아래 전시회 찾기 실패: '{user_input}'")
    inputContentIndex = matchingRows.index[0]
    inputContentEmbedding = contentsEmbeddings[inputContentIndex]
    similarities = util.cos_sim(inputContentEmbedding, contentsEmbeddings).flatten()
    if len(similarities) < topN:
        raise ValueError("비교할 데이터가 부족")
    similarIndices = (-similarities).argsort()[:min(topN, len(similarities))]
    similarExhibitions = myExiDf.loc[similarIndices][['exhibition_title', 'exhibition_img']].values.tolist()
    return similarExhibitions

