from flask import Flask, request
from flask_cors import CORS, cross_origin

from indexing import build_save_vectorizer
from crawling import crawl_dataset

app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False


@app.route('/build-vectorizer', methods=['GET'])
@cross_origin()
def build_vectorizer():
    dataset = request.args.get('dataset')
    build_save_vectorizer(dataset)
    return 'true'

@app.route('/crawl', methods=['GET'])
@cross_origin()
def build_vectorizer():
    dataset = request.args.get('dataset')
    crawl_dataset(dataset)
    return 'true'


if __name__ == "__main__":
    app.run(port=8001, debug=True)
