from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import base64
from werkzeug.utils import secure_filename
import sys
import tempfile
import time
from datetime import datetime

# Import our image processing module
from image_processing import process_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Operation history will be stored here (in-memory storage)
# In a production environment, you might want to use a database
operations_history = []

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
            # Clean up the base64 data to remove the data URL prefix if present
            image2_data = data['image2'].split(',')[1] if ',' in data['image2'] else data['image2']
            
            # Debug information
            print(f"Image2 present, operation: {operation}")
            
            if isinstance(operation, list):
                # Her işlem için ikinci görüntüyü ekle
                for i in range(len(operation)):
                    if params[i] is None:
                        params[i] = {}
                    params[i]['img2_data'] = image2_data
            else:
                # For single operation mode
                if params is None:
                    params = {}
                params['img2_data'] = image2_data
        
        # Debug info
        print(f"Processing with operation: {operation}")
        if isinstance(operation, list):
            for i, op in enumerate(operation):
                if op in ['add_images', 'divide_images']:
                    print(f"Operation {op} has img2_data: {'img2_data' in params[i]}")
        elif operation in ['add_images', 'divide_images']:
            print(f"Operation {operation} has img2_data: {'img2_data' in params}")
        
        # Process the image
        start_time = time.time()
        result = process_image(image_data, operation, params)
        processing_time = time.time() - start_time
        
        # Record the operation in history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'processing_time': round(processing_time, 2),
            'success': True
        }
        operations_history.append(history_entry)
        
        # Limit history size
        if len(operations_history) > 50:  # Keep last 50 operations
            operations_history.pop(0)
            
        # Include the history in the response
        result['history'] = operations_history
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Record failed operation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'operation': data.get('operation', 'unknown'),
            'error': str(e),
            'success': False
        }
        operations_history.append(history_entry)
        
        return jsonify({'error': str(e), 'history': operations_history}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Endpoint to get operation history"""
    return jsonify({'history': operations_history})

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
        
        # Record upload in history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'operation': 'upload',
            'filename': filename,
            'success': True
        }
        operations_history.append(history_entry)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': f'/static/uploads/{filename}',
            'base64': f'data:image/jpeg;base64,{base64_data}',
            'history': operations_history
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 