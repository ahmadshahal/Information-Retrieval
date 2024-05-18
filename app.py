import threading

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from collections import OrderedDict

from match_and_rank import match_and_rank
from query_correction import correct_query

# from search_query_processing import get_search_result
from text_preprocessing import get_preprocessed_text_terms

app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False


# @app.route('/search', methods=['GET'])
# @cross_origin()
# def search():
#     query = request.args.get('query')
#     dataset = request.args.get('dataset')
#     retrieving_relevant_on = "terms"  # "terms" or "topics"
#     return get_search_result(query, dataset, retrieving_relevant_on)


@app.route('/process-text', methods=['GET'])
@cross_origin()
def get_processed_text():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    return get_preprocessed_text_terms(text, dataset)


@app.route('/correct-query', methods=['GET'])
@cross_origin()
def fix_query():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    corrected_query = correct_query(text)
    return match_and_rank(corrected_query, dataset)


@app.route('/ranking', methods=['GET'])
@cross_origin()
def get_ranking():
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    return match_and_rank(query, dataset)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
