from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model and vectorizer
model = joblib.load('classification_model-2')  # Update with your model path
tokenizer = joblib.load('tokenizer')  # Update with your vectorizer path

@app.route('/classify', methods=['POST'])
def classify_comment():
    try:
        data = request.json
        comment = data.get('comment', '')
        tokenized_comment = tokenizer.transform([comment])
        prediction = model.predict(tokenized_comment)
        result = "Cyberbullying" if prediction[0] == 1 else "Not Cyberbullying"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensure it's running on port 5000