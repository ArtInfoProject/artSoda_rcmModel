# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from rcmForFlask import preprocess_data, recommend

app = Flask(__name__)
CORS(app)
myExiDf, contentsEmbeddings = preprocess_data()


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_input = request.json.get('exhibition_title')
    print(f"통신받은 전시회 이름: {user_input}")
    similar_exhibitions = recommend(user_input, myExiDf, contentsEmbeddings)
    response_data = [{'exhibition_title': i[0], 'exhibition_img': i[1]} for i in similar_exhibitions[1:]]
    print('비슷한 전시회:')
    for data in response_data:
        print(data)
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(port=5002)
