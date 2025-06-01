from flask import Flask, request, jsonify, render_template, send_file
import qrcode
import io
import base64
from PIL import Image
import cv2
import numpy as np
import os
import urllib.parse
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure static directories exist
os.makedirs('static/uploads', exist_ok=True)


def validate_url(url):
    """Validate URL format."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_qr():
    """Generate QR code from URL."""
    try:
        data = request.json
        url = data.get('url', '').strip()
        size = data.get('size', 'medium')
        format_type = data.get('format', 'png').lower()

        # Validate input
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'})

        if not validate_url(url):
            return jsonify({'success': False, 'error': 'Invalid URL format'})

        # Size mapping
        size_map = {
            'small': 8,
            'medium': 10,
            'large': 15
        }
        box_size = size_map.get(size, 10)

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Create image
        qr_image = qr.make_image(fill_color="black", back_color="white")

        # Convert to base64 for web display
        buffer = io.BytesIO()

        if format_type == 'jpeg':
            # Convert to RGB for JPEG
            qr_image = qr_image.convert('RGB')
            qr_image.save(buffer, format='JPEG', quality=95)
            mime_type = 'image/jpeg'
        else:
            # Default to PNG
            qr_image.save(buffer, format='PNG')
            mime_type = 'image/png'

        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': f'data:{mime_type};base64,{img_base64}',
            'format': format_type,
            'size': size,
            'url': url
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/scan', methods=['POST'])
def scan_qr():
    """Scan QR code from uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        # Read image file
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({'success': False, 'error': 'Empty file'})

        # Convert to OpenCV format
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})

        # Detect QR code
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)

        if data:
            # Check if detected data is a URL
            is_url = validate_url(data)

            return jsonify({
                'success': True,
                'data': data,
                'is_url': is_url,
                'bbox': bbox.tolist() if bbox is not None else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No QR code found in the image'
            })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/download/<format_type>')
def download_qr(format_type):
    """Generate and download QR code file."""
    try:
        url = request.args.get('url')
        size = request.args.get('size', 'medium')

        if not url or not validate_url(url):
            return jsonify({'error': 'Invalid URL'}), 400

        # Generate QR code
        size_map = {'small': 8, 'medium': 10, 'large': 15}
        box_size = size_map.get(size, 10)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        qr_image = qr.make_image(fill_color="black", back_color="white")

        # Save to buffer
        buffer = io.BytesIO()

        if format_type.lower() == 'jpeg':
            qr_image = qr_image.convert('RGB')
            qr_image.save(buffer, format='JPEG', quality=95)
            mimetype = 'image/jpeg'
            extension = 'jpg'
        else:
            qr_image.save(buffer, format='PNG')
            mimetype = 'image/png'
            extension = 'png'

        buffer.seek(0)

        # Generate filename
        clean_url = urllib.parse.urlparse(url).netloc.replace('.', '_')
        filename = f'qr_code_{clean_url}.{extension}'

        return send_file(
            buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'QR Code Generator API is running'})


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)