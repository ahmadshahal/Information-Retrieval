from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from text_preprocessing import get_preprocessed_text_terms

app = Flask(__name__)
cors = CORS(app)
app.json.sort_keys = False


@app.route('/process-text', methods=['GET'])
@cross_origin()
def get_processed_text():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    return jsonify(get_preprocessed_text_terms(text, dataset))


if __name__ == "__main__":
    app.run(port=8000, debug=True)
