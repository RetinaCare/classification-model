from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_type(file_stream):
    """Validate that the file is actually a JPEG image"""
    try:
        file_stream.seek(0)
        
        img = Image.open(file_stream)
        
        # Check if format is JPEG
        is_jpeg = img.format == 'JPEG'
        
        file_stream.seek(0)
        
        return is_jpeg
    except Exception:
        file_stream.seek(0)
        return False


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if image file is present in request
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'File must be a JPEG or JPG image'
        }), 400
    
    # Validate actual file type
    if not validate_image_type(file.stream):
        return jsonify({
            'success': False,
            'error': 'File is not a valid JPEG image'
        }), 400
    
    # If all validations pass
    filename = secure_filename(file.filename)
    return jsonify({
        'success': True,
        'message': 'Image validated successfully',
        'filename': filename
    }), 200

# Test route to upload form
@app.route('/', methods=['GET'])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Upload JPEG/JPG Image</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept=".jpg,.jpeg" required>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
