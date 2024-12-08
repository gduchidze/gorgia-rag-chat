from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot1 import create_gorgia_agent
from chatbot1 import logger

app = Flask(__name__)
CORS(app)

try:
    agent = create_gorgia_agent()
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {str(e)}")
    raise


def format_response_for_frontend(response: dict) -> dict:
    try:
        if not response:
            return {
                'response': {
                    'type': 'text',
                    'content': 'No response from agent'
                }
            }
        if isinstance(response, dict) and response.get('type') == 'product_list':
            return {
                'response': {
                    'type': 'product_list',
                    'content': response.get('message', ''),
                    'products': response.get('products', [])
                }
            }
        if isinstance(response, dict) and (response.get('type') == 'text' or 'message' in response):
            return {
                'response': {
                    'type': 'text',
                    'content': response.get('message', str(response))
                }
            }

        if isinstance(response, dict) and response.get('type') == 'error':
            return {
                'error': response.get('message', 'An unknown error occurred')
            }

        if isinstance(response, str):
            return {
                'response': {
                    'type': 'text',
                    'content': response
                }
            }

        logger.warning(f"Unexpected response format: {response}")
        return {
            'response': {
                'type': 'text',
                'content': str(response) if response else 'Empty response'
            }
        }

    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return {
            'error': 'Error formatting response'
        }


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided'
            }), 400

        message = data['message']
        logger.info(f"Received message: {message}")

        response = agent.run(message)
        logger.info(f"Raw agent response: {response}")

        formatted_response = format_response_for_frontend(response)
        logger.info(f"Formatted response: {formatted_response}")

        if 'response' in formatted_response:
            content = formatted_response['response'].get('content')
            if not isinstance(content, str):
                formatted_response['response']['content'] = str(content)

        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': {
                'type': 'text',
                'content': f"Error processing request: {str(e)}"
            }
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    try:
        agent.clear_history()
        return jsonify({
            'response': {
                'type': 'text',
                'content': 'Chat history cleared'
            }
        })
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        if agent:
            return jsonify({
                'status': 'healthy',
                'agent_status': 'initialized'
            }), 200
        return jsonify({
            'status': 'degraded',
            'message': 'Agent not initialized'
        }), 503
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)