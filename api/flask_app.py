import io
import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SRC = os.path.join(BASE_DIR, "model", "src")
sys.path.append(MODEL_SRC)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from fusion_model import FusionModel

# * ******************************************************
# App Configs
# * ******************************************************
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

# * ******************************************************
# Endpoints
# * ******************************************************
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        filename = validate_request()
        return jsonify({
            'success': True,
            'message': 'Image validated successfully',
            'filename': filename
        }), 200

    except Exception as e:
        return jsonify({
            'success': True,
            'message': repr(e)
        }), 400


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

# * ******************************************************
#  HELPER FUNCTIONS
# * ******************************************************
def validate_request():
    if 'image' not in request.files:
        raise Exception("No image file provided")

    file = request.files['image']
    if file.filename == '':
        raise Exception("No file selected")

    if not allowed_file(file.filename):
        raise Exception("File must be a JPEG or JPG image")

    if not validate_image_type(file.stream):
        raise Exception("File is not a valid JPEG image")

    return secure_filename(file.filename)


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_type(file_stream):
    """Validate that the file is actually a JPEG image"""
    try:
        file_stream.seek(0)
        img = Image.open(file_stream)
        is_jpeg = img.format == 'JPEG'
        file_stream.seek(0)
        return is_jpeg
    except Exception:
        file_stream.seek(0)
        return False


def load_model():
    """
    Loads classfication model from Digital Ocean Spaces.
    """
    try:
        url = "https://s3.retinacare.ams3.digitaloceanspaces.com/fusion_model_mvp.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model = FusionModel().to("cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return False


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
