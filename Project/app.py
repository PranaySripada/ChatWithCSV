import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI
import logging

app = Flask(__name__)
CORS(app)

# Set your OpenAI API Key securely
os.environ["OPENAI_API_KEY"] = "ENTER_YOUR_SECRET_API_KEY"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

agent_executor = None
csv_file_path = None
logging.basicConfig(level=logging.DEBUG)

def create_agent(file_path):
    global agent_executor, csv_file_path
    try:
        df = pd.read_csv(file_path)  # Checking if CSV can be read
        agent_executor = create_csv_agent(OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"]),
                                          file_path,
                                          verbose=True, allow_dangerous_code=True)
        csv_file_path = file_path
        logging.info(f'Agent created for {file_path}')
    except Exception as e:
        logging.error(f'Failed to create agent: {e}')
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            logging.info(f'File saved to {file_path}')
            create_agent(file_path)
            return jsonify({'success': True})
        except Exception as e:
            logging.error(f'Error processing file: {e}')
            return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    if agent_executor is None:
        logging.error('No CSV file uploaded or agent not created')
        return jsonify({'error': 'No CSV file uploaded or agent not created'}), 400
    
    user_input = request.json.get('input', '')
    try:
        response = agent_executor.invoke(user_input)
        output_text = response.get('output', 'No output generated')

        plot_data = None

        if "plot" in user_input.lower():
            df = pd.read_csv(csv_file_path)
            plt.figure(figsize=(10, 5))
            df.plot(kind='line')  # Customize the plot as needed
            plt.title('Sample Plot')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

        return jsonify({
            'output': output_text,
            'plot_data': plot_data
        })

    except Exception as e:
        logging.error(f'Error during query execution: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
