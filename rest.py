from flask import Flask, request, jsonify

import constants
import main

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def query():

    try:
        data = request.get_json()

        #Error handling
        if 'prompt' in data and data['prompt'] == "":
            return jsonify({'error': 'Empty prompt'}), 400
        
        if 'prompt' not in data and ('init' not in data or 'init' in data and not data['init']):
            return jsonify({'error': 'Missing "prompt" element in request body'}), 400

        if 'prompt' in data and 'init' in data and data['init']:
            return jsonify({'error': 'Cannot initialize and respond to prompt at the same time'}), 400

        if 'prompt' in data and 'validate' in data and data['validate']:
            return jsonify({'error': 'Cannot validate test prompts and respond to user prompt at the same time'}), 400
        
        if 'init' in data and data['init'] and 'validate' in data and data['validate']:
            return jsonify({'error': 'Cannot initialize and validate test prompts at the same time'}), 400

        #Read json
        args = constants.DefaultArgs
        if 'prompt' in data:
            args.question = data['prompt']
        if 'database' in data:
            args.database = data['database']
        if 'model' in data:
            args.model = data['model']
        if 'init' in data:
            args.init = data['init']
        if 'validate' in data:
            args.init = data['validate']
        if 'cv' in data:
            args.cv = data['cv']
        
        #Invoke chain
        result = main.invoke_chain(args)

        return jsonify({'response': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)