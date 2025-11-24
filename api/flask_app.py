import io
import sys
import os
import torch

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# * ******************************************************
# Due to how python works, I need to load from paths
# this way.
# * ******************************************************
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SRC = os.path.join(BASE_DIR, "model", "src")
sys.path.append(MODEL_SRC)

from fusion_model import FusionModel

# * ******************************************************
# App Configs
# * ******************************************************
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

# * ******************************************************
# Model loading
# * ******************************************************
def load_model():
    """
    Loads classification model from Digital Ocean Spaces.
    """
    try:
        url = "https://s3.retinacare.ams3.digitaloceanspaces.com/fusion_model_mvp.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            map_location="cpu",
            check_hash=False,
            progress=True
        )
        model = FusionModel().to("cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return False

model = load_model()

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
            'success': False,
            'message': str(e)
        }), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        clin_features = validate_clinical_features()
        return jsonify({
            'success': True,
            'message': "Valid request"
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
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
        file_stream.seek(0)
        return img.format == 'JPEG'
    except Exception:
        file_stream.seek(0)
        return False


def validate_clinical_features():
    if request.json is None:
        raise Exception("Clinical features not provided")

    if request.json.get("hba1c") is None:
        raise Exception("Hba1c key is missing")

    if request.json.get("blood_pressure") is None:
        raise Exception("Blood Pressure key is missing")

    if request.json.get("duration") is None:
        raise Exception("Duration key is missing")

    return request.json


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
