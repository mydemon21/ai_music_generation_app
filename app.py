import os
import uuid
import logging
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Import configuration
from config import Config

# Use real AI models for music generation
Config.USE_MOCK_MODELS = False

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload and generated directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif'},
    'audio': {'wav', 'mp3', 'ogg'}
}

# Initialize our generators (lazy loading)
text_to_music_generator = None
image_to_music_generator = None
audio_to_music_generator = None

def get_text_to_music_generator():
    """Get or initialize the text-to-music generator."""
    global text_to_music_generator
    if text_to_music_generator is None:
        logger.info(f"Initializing TextToMusicGenerator with model size: {Config.MODEL_SIZE}")
        from models.text_to_music import TextToMusicGenerator
        text_to_music_generator = TextToMusicGenerator(model_size=Config.MODEL_SIZE)
    return text_to_music_generator

def get_image_to_music_generator():
    """Get or initialize the image-to-music generator."""
    global image_to_music_generator
    if image_to_music_generator is None:
        logger.info("Initializing ImageToMusicGenerator")
        from models.image_to_music import ImageToMusicGenerator
        image_to_music_generator = ImageToMusicGenerator()
    return image_to_music_generator

def get_audio_to_music_generator():
    """Get or initialize the audio-to-music generator."""
    global audio_to_music_generator
    if audio_to_music_generator is None:
        logger.info("Initializing AudioToMusicGenerator")
        from models.audio_to_music import AudioToMusicGenerator
        audio_to_music_generator = AudioToMusicGenerator()
    return audio_to_music_generator

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-from-text', methods=['POST'])
def generate_from_text():
    if 'text_prompt' not in request.form:
        return jsonify({'error': 'No text prompt provided'}), 400

    text_prompt = request.form['text_prompt']
    logger.info(f"Received text prompt: {text_prompt}")

    # Get optional parameters
    genre = request.form.get('genre')
    tempo = request.form.get('tempo')
    mood = request.form.get('mood')

    # Generate a unique filename for the output
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

    try:
        # Get the generator
        generator = get_text_to_music_generator()

        start_time = time.time()

        # Generate music using the AI model
        generator.generate(
            text_prompt=text_prompt,
            output_path=output_path,
            duration=Config.DEFAULT_DURATION,
            genre=genre,
            tempo=tempo,
            mood=mood
        )

        generation_time = time.time() - start_time

        logger.info(f"Music generation completed in {generation_time:.2f} seconds")

        return jsonify({
            'success': True,
            'message': f'Music generated from text: "{text_prompt}"',
            'audio_path': os.path.join('static', 'generated', output_filename),
            'generation_time': f"{generation_time:.2f} seconds"
        })
    except Exception as e:
        logger.error(f"Error generating music from text: {str(e)}")
        return jsonify({'error': f'Failed to generate music: {str(e)}'}), 500

@app.route('/generate-from-image', methods=['POST'])
def generate_from_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        # Generate a unique filename to prevent collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        logger.info(f"Image saved to {file_path}")

        # Get optional parameters
        genre = request.form.get('genre')
        tempo = request.form.get('tempo')
        mood_influence = request.form.get('mood')

        # Generate a unique filename for the output
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

        try:
            # Get the generator
            generator = get_image_to_music_generator()

            start_time = time.time()

            # Generate music using the AI model
            result = generator.generate(
                image_path=file_path,
                output_path=output_path,
                duration=Config.DEFAULT_DURATION,
                genre=genre,
                tempo=tempo,
                mood_influence=mood_influence
            )

            generation_time = time.time() - start_time

            logger.info(f"Music generation from image completed in {generation_time:.2f} seconds")

            # Extract image analysis for the response
            mood_analysis = result.get('mood_analysis', {})
            image_description = result.get('image_description', 'Image analyzed successfully')

            return jsonify({
                'success': True,
                'message': f'Music generated from image: "{filename}"',
                'image_path': os.path.join('static', 'uploads', unique_filename),
                'audio_path': os.path.join('static', 'generated', output_filename),
                'generation_time': f"{generation_time:.2f} seconds",
                'image_description': image_description,
                'mood_analysis': {
                    'mood_score': mood_analysis.get('mood_score', 50),
                    'dominant_color': mood_analysis.get('dominant_color', 'neutral'),
                    'brightness': mood_analysis.get('brightness', 128)
                }
            })
        except Exception as e:
            logger.error(f"Error generating music from image: {str(e)}")
            return jsonify({'error': f'Failed to generate music: {str(e)}'}), 500

    return jsonify({'error': 'Invalid image format'}), 400

@app.route('/generate-from-audio', methods=['POST'])
def generate_from_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio provided'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No audio selected'}), 400

    if file and allowed_file(file.filename, 'audio'):
        filename = secure_filename(file.filename)
        # Generate a unique filename to prevent collisions
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        logger.info(f"Audio saved to {file_path}")

        # Get optional parameters
        genre = request.form.get('genre')
        tempo_adjustment = request.form.get('tempo')
        complexity = request.form.get('complexity')

        # Generate a unique filename for the output
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)

        try:
            # Get the generator
            generator = get_audio_to_music_generator()

            start_time = time.time()

            # Generate music using the AI model
            result = generator.generate(
                audio_path=file_path,
                output_path=output_path,
                duration=Config.DEFAULT_DURATION,
                genre=genre,
                tempo_adjustment=tempo_adjustment,
                complexity=complexity
            )

            generation_time = time.time() - start_time

            logger.info(f"Music generation from audio completed in {generation_time:.2f} seconds")

            # Extract melody features for the response
            melody_features = result.get('melody_features', {})
            generated_prompt = result.get('generated_prompt', 'Audio analyzed successfully')

            return jsonify({
                'success': True,
                'message': f'Music generated from audio: "{filename}"',
                'original_audio_path': os.path.join('static', 'uploads', unique_filename),
                'generated_audio_path': os.path.join('static', 'generated', output_filename),
                'generation_time': f"{generation_time:.2f} seconds",
                'generated_prompt': generated_prompt,
                'melody_analysis': {
                    'tempo': melody_features.get('tempo', 90),
                    'is_major': melody_features.get('is_major', True),
                    'rhythm_regularity': melody_features.get('rhythm_regularity', 0.5)
                }
            })
        except Exception as e:
            logger.error(f"Error generating music from audio: {str(e)}")
            return jsonify({'error': f'Failed to generate music: {str(e)}'}), 500

    return jsonify({'error': 'Invalid audio format'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
