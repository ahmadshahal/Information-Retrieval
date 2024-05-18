import threading

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from collections import OrderedDict

from match_and_rank import match_and_rank
from query_correction import spell_check_query

from corpus import get_corpus

from text_preprocessing import get_preprocessed_text_terms

app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False


@app.route('/process-text', methods=['GET'])
@cross_origin()
def get_processed_text():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    return get_preprocessed_text_terms(text, dataset)


@app.route('/correct-query', methods=['GET'])
@cross_origin()
def correct_query():
    text = request.args.get('query')
    dataset = request.args.get('dataset')
    corrected_query = spell_check_query(text)
    queries = get_corpus(dataset)
    quereis_ids = match_and_rank(corrected_query, dataset)
    filtered = dict(filter(lambda item: item[0] in quereis_ids, queries.items()))
    return filtered


@app.route('/ranking', methods=['GET'])
@cross_origin()
def get_ranking():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    docs = get_corpus(dataset)
    docs_ids = match_and_rank(query, dataset).keys()
    filtered = dict(filter(lambda item: item[0] in docs_ids, docs.items()))
    return filtered


if __name__ == "__main__":
    app.run(port=8000, debug=True)
