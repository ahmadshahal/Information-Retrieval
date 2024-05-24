from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from match_and_rank import match_and_rank
from query_correction import process_query
app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False

from libs import corpus

@app.route('/correct-query', methods=['GET'])
@cross_origin()
def correct_query():
    text = request.args.get('query')
    dataset = request.args.get('dataset')
    processed_query = process_query(text)
    queries = corpus.get_corpus(dataset)
    quereis_ids = match_and_rank(processed_query, dataset)
    filtered = {}
    for id in quereis_ids:
        filtered[id] = queries[id]
    return filtered


@app.route('/ranking', methods=['GET'])
@cross_origin()
def get_ranking():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    docs = corpus.get_corpus(dataset)
    docs_ids = match_and_rank(query, dataset).keys()
    filtered = {}
    for id in docs_ids:
        filtered[id] = docs[id]
    return filtered


if __name__ == "__main__":
    app.run(port=8002, debug=True)
