# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import shutil # For checking command existence

# Load environment variables from .env file
basedir = Path(__file__).resolve().parent.parent
load_dotenv(basedir / '.env')

class Config:
    # --- General Flask & App Config ---
    SECRET_KEY = os.environ.get('SECRET_KEY', 'insecure-default-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    HOST = "0.0.0.0"
    PORT = int(os.environ.get('PORT', 5000))
    BASE_DIR = basedir

    # --- Logging ---
    LOGS_DIR = BASE_DIR / "logs"
    LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

    # --- Database ---
    DB_HOST = os.environ.get('DB_HOST')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_NAME = os.environ.get('DB_NAME')
    DB_POOL_NAME = "app_pool"
    DB_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', 5))

    # --- ASR/EMR Service ---
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None
    GEMINI_HEADERS = {'Content-Type': 'application/json'}
    GEMINI_CONFIG = {"temperature": 0.3, "topP": 0.8, "topK": 40}
    GEMINI_TIMEOUT_SECONDS = 60

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB for uploads
    ALLOWED_AUDIO_MIMETYPES = {'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/webm'}
    ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.mpeg', '.webm'}
    ALLOWED_IMAGE_MIMETYPES = {'image/jpeg', 'image/png', 'image/tiff', 'image/bmp', 'image/webp'}
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    TARGET_SAMPLE_RATE = 16000
    FFMPEG_PATH = os.environ.get('FFMPEG_PATH', "ffmpeg")
    CHUNK_SIZE_MS = 30000 # For language detection sample

    # --- OCR Service ---
    TESSERACT_CMD = os.environ.get('TESSERACT_CMD') # Path can be set in env or auto-detected by pytesseract
    OCR_UPLOAD_FOLDER = BASE_DIR / 'uploads'

    # --- ML Models Paths ---
    PICKLE_MODELS_DIR = BASE_DIR / "saved_models"
    HF_MODELS_DIR = BASE_DIR / "models" # Cache directory

    # --- CORS ---
    CORS_ORIGINS = "*" # Or specify origins: ["http://localhost:3000", "https://yourfrontend.com"]

    # --- SocketIO ---
    # Explicitly set to None to use default (threading/werkzeug based) async mode.
    # Remove this line entirely also works.
    SOCKETIO_ASYNC_MODE = None


    @classmethod
    def validate(cls):
        """Validate critical configurations."""
        if cls.SECRET_KEY == 'insecure-default-key':
             print("WARNING: SECRET_KEY is insecure. Set a strong secret key in .env")
        if not all([cls.DB_HOST, cls.DB_USER, cls.DB_PASSWORD, cls.DB_NAME]):
            raise ValueError("CRITICAL: Database config missing in .env")
        if not cls.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY missing. ASR/EMR features may fail.")
            cls.GEMINI_API_URL = None
        if shutil.which(cls.FFMPEG_PATH) is None:
             print(f"WARNING: FFmpeg ('{cls.FFMPEG_PATH}') not found. Audio conversion will fail.")
        try:
            import pytesseract
            if cls.TESSERACT_CMD:
                 pytesseract.pytesseract.tesseract_cmd = cls.TESSERACT_CMD
            version = pytesseract.get_tesseract_version()
            print(f"INFO: Found Tesseract version {version}.")
        except Exception as e:
            print(f"WARNING: Tesseract not found or configured correctly ({e}). OCR will fail. Ensure Tesseract is installed and in PATH, or set TESSERACT_CMD in .env.")

    @classmethod
    def create_directories(cls):
        """Creates necessary directories."""
        for dir_path in [cls.LOGS_DIR, cls.PICKLE_MODELS_DIR, cls.HF_MODELS_DIR, cls.OCR_UPLOAD_FOLDER]:
            dir_path.mkdir(parents=True, exist_ok=True)

# --- Initial Setup ---
Config.create_directories()
Config.validate()