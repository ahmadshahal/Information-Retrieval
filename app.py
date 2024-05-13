import threading

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from collections import OrderedDict


from inverted_index_store import set_inverted_index_store_global_variables, get_weighted_inverted_index, \
    create_weighted_inverted_index, get_document_vector, get_documents_vector
from matching_and_ranking import ranking
from query_processing import calculate_query_tfidf
from search_query_processing import get_search_result
from text_preprocessing import get_preprocessed_text_terms

app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False


@app.route('/search', methods=['GET'])
@cross_origin()
def search():
    query = request.args.get('query')
    dataset = request.args.get('dataset')
    retrieving_relevant_on = "terms"  # "terms" or "topics"
    return get_search_result(query, dataset, retrieving_relevant_on)


@app.route('/process-text', methods=['GET'])
@cross_origin()
def get_processed_text():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    return get_preprocessed_text_terms(text, dataset)

@app.route('/ranking', methods=['GET'])
@cross_origin()
def get_ranking() -> OrderedDict[str, float]:
    dataset = request.args.get('dataset')
    query = request.args.get('query')
    return ranking(query, dataset)


if __name__ == "__main__":
    set_inverted_index_store_global_variables()
    app.run(port=8000, debug=True)
