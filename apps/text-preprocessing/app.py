from flask import Flask, request
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
    return get_preprocessed_text_terms(text, dataset)

@app.route('/hi', methods=['GET'])
@cross_origin()
def hi():
    text = request.args.get('text')
    dataset = request.args.get('dataset')
    return f"hello {text} from {dataset}"


if __name__ == "__main__":
    app.run(port=8000, debug=True)
