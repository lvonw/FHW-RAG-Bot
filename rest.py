from flask import Flask, request, jsonify, send_from_directory, redirect
#from flask_cors import CORS

import constants
import main
from parsing import EnumAction

app = Flask(__name__)

#Cors Einstellung wird für den Frontend Developer-Server benötigt
#cors = CORS(app, resources={r"/chat": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["GET"])
def redirect_internal():
    return redirect("/docs/", code=302)


@app.route('/docs/')
def index():
  return send_from_directory("docs" , 'index.html')

@app.route('/docs/<path:path>')
def send_docs(path):
    return send_from_directory('docs', path)
    
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
            EnumAction(type=constants.LoaderMethod, option_strings="", dest="database")(None, args,values=[data['database']])
        if 'model' in data:
            EnumAction(type=constants.ModelMethod, option_strings="", dest="model")(None, args, values=[data['model']])
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