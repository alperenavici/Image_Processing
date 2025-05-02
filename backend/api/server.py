from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import base64
from werkzeug.utils import secure_filename
import sys
import tempfile

# Import our image processing module
from image_processing import process_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

@app.route('/api/process', methods=['POST'])
def process():
    try:
        data = request.json
        
        if not data or 'image' not in data or 'operation' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get base64 image data, operation name, and parameters
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        operation = data['operation']  # Bu artık bir string veya string listesi olabilir
        params = data.get('params', {})
        
        # Birden fazla işlemi kontrol et
        if isinstance(operation, list):
            # params bir liste değilse, her işlem için aynı parametreleri kullanır
            if not isinstance(params, list):
                params = [params] * len(operation)
            elif len(params) < len(operation):
                # Eksik parametreleri None ile doldur
                params.extend([None] * (len(operation) - len(params)))
                
        # If we have a second image for operations that need two images
        if 'image2' in data:
            image2_data = data['image2'].split(',')[1] if ',' in data['image2'] else data['image2']
            if isinstance(operation, list):
                # Her işlem için ikinci görüntüyü ekle
                for i in range(len(operation)):
                    if params[i] is None:
                        params[i] = {}
                    params[i]['img2_data'] = image2_data
            else:
                if params is None:
                    params = {}
                params['img2_data'] = image2_data
        
        # Process the image
        result = process_image(image_data, operation, params)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read the file and convert to base64
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        base64_data = base64.b64encode(file_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/static/uploads/{filename}',
            'base64': f'data:image/jpeg;base64,{base64_data}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 