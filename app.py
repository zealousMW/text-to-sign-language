from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
from pathlib import Path
import logging
from werkzeug.utils import safe_join
import mimetypes

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure folders
app.config['UPLOAD_FOLDER'] = Path('output').absolute()
app.config['VIDEO_FOLDER'] = Path('video').absolute()

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['VIDEO_FOLDER'].mkdir(exist_ok=True)

# Initialize translator
from main3 import ISLTranslator
translator = ISLTranslator(
    video_directory=str(app.config['VIDEO_FOLDER']),
    output_directory=str(app.config['UPLOAD_FOLDER'])
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        sentence = data.get('sentence', '')
        
        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400
            
        # Get the processed words in SOV order
        words = translator.extract_sov(sentence)
        logger.info(f"Extracted words: {words}")
        
        # Translate to video
        success = translator.translate_to_video(sentence)
        
        if success:
            # Get the latest video file
            video_files = list(app.config['UPLOAD_FOLDER'].glob('*.mp4'))
            latest_video = max(video_files, key=os.path.getctime)
            video_filename = latest_video.name
            
            return jsonify({
                'success': True,
                'words': words,
                'video_url': f'/video/{video_filename}'
            })
        else:
            return jsonify({'error': 'Translation failed'}), 500
            
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>')
def get_video(filename):
    try:
        video_path = safe_join(str(app.config['UPLOAD_FOLDER']), filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
            
        # Set the correct MIME type for MP4 videos
        response = send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=filename
        )
        
        # Add headers to help with video playback
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        return response
        
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)