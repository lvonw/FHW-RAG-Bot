from flask import Flask, request, jsonify

import constants
import main

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def query():

    try:
        data = request.get_json()

        if 'prompt' not in data:
            return jsonify({'error': 'Missing "prompt" element in request body'}), 400

        args = constants.DefaultArgs
        args.question = data['prompt']
        
        result = main.invoke_chain(args)

        return jsonify({'response': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)