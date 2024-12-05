import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import create_gorgia_agent

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://localhost:3001", "https://gorgia-rag-front.vercel.app/"]}})
CORS(app)

agent = create_gorgia_agent()


def format_response(response):
    cleaned_response = re.sub(r'<tool>.*?</tool>', '', response).strip()
    if not cleaned_response and 'search_' in response:
        match = re.search(r'<tool>search_\w+\|(.*?)</tool>', response)
        if match:
            return match.group(1)

    return cleaned_response or response


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        response = agent.run(message)

        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    try:
        agent.clear_history()
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
