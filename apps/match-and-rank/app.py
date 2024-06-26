from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from waitress import serve


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
    queries_ids = match_and_rank(processed_query, dataset)
    items = [{"id": query_id, "text": queries[query_id]} for query_id in queries_ids]
    return {
        "status": "success",
        "code": 200,
        "data": items
    }


@app.route('/ranking', methods=['GET'])
@cross_origin()
def get_ranking():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    docs = corpus.get_corpus(dataset)
    results = match_and_rank(query, dataset)
    docs_ids = results.keys()
    items = [{"id": doc_id, "text": docs[doc_id], "value": results[doc_id]} for doc_id in docs_ids]
    return {
        "status": "success",
        "code": 200,
        "data": items
    }

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=8002)
