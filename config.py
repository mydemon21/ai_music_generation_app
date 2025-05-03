import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    # App settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    GENERATED_FOLDER = os.getenv('GENERATED_FOLDER', 'static/generated')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB default

    # Model settings
    USE_MOCK_MODELS = os.getenv('USE_MOCK_MODELS', 'False').lower() in ('true', '1', 't')
    MODEL_SIZE = os.getenv('MODEL_SIZE', 'small')  # Options: small, medium, large

    # Generation settings
    DEFAULT_DURATION = float(os.getenv('DEFAULT_DURATION', 10.0))

    # Debug settings
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'

    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')

    # API rate limiting
    RATE_LIMIT = int(os.getenv('RATE_LIMIT', 10))  # Requests per minute
