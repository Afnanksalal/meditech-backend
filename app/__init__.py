# app/__init__.py
import os
import logging
from flask import Flask, jsonify, g, request
import json
from flask_socketio import SocketIO
from flask_cors import CORS
from werkzeug.exceptions import HTTPException


# Import config and modules
from .config import Config
from .database import init_db_pool, create_tables_if_not_exist, close_db_connection, get_db_connection
from .models import load_pickle_models, load_hf_models

# --- Global Variables (Initialized in create_app) ---
# SocketIO instance
socketio = SocketIO()
# In-memory store for active rooms (consider Redis for multi-process/scaled deployment)
active_rooms = {}

def create_app(config_class=Config):
    """Application Factory Function"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # --- Initialize Logging ---
    # Logging setup is done globally when config is imported,
    # but we can access the logger via app.logger
    app.logger.info(f"Flask app '{app.name}' created.")
    app.logger.info(f"Debug mode: {app.debug}")

    # --- WebRTC Configuration ---
    # Define STUN servers here - make it configurable (e.g., via environment variables) if needed
    app.config['WEBRTC_CONFIG'] = {
        'iceServers': [
            {'urls': 'stun:stun.l.google.com:19302'},
            {'urls': 'stun:stun1.l.google.com:19302'},
            { 'urls': 'stun:stun2.l.google.com:19302' },
            # Add more STUN/TURN servers if required
        ]
    }

    # --- Initialize Extensions ---
    CORS(app, resources={r"/*": {"origins": config_class.CORS_ORIGINS}})
    # Pass async_mode from config
    socketio.init_app(app, cors_allowed_origins="*", async_mode=config_class.SOCKETIO_ASYNC_MODE)
    app.logger.info(f"SocketIO initialized with async_mode='{config_class.SOCKETIO_ASYNC_MODE}'")

    # --- Initialize Database ---
    with app.app_context():
        init_db_pool(app.config) # Initialize pool using app config
        create_tables_if_not_exist() # Create tables if needed

    # --- Load Machine Learning Models ---
    with app.app_context():
        app.logger.info("Loading prediction models...")
        load_pickle_models()
        app.logger.info("Loading ASR models...")
        load_hf_models()
        app.logger.info("Model loading process initiated.")


    # --- Register Blueprints ---
    from .routes.asr_routes import asr_bp
    from .routes.predict_routes import predict_bp
    from .routes.ocr_routes import ocr_bp
    from .routes.suggestions_routes import suggestions_bp

    app.register_blueprint(asr_bp, url_prefix='/api') # Add prefix if desired
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(ocr_bp, url_prefix='/api')
    app.register_blueprint(suggestions_bp, url_prefix='/api')
    app.logger.info("Blueprints registered.")

    # --- Register SocketIO Handlers ---
    # Import handlers AFTER socketio is initialized and app context might be needed
    from . import sockets # This registers the handlers defined in sockets.py
    app.logger.info("SocketIO event handlers registered.")


    # --- Teardown Context ---
    # Although using pooling, this ensures connections aren't held open
    # unnecessarily if g was used (we are not using g for DB conn now)
    # @app.teardown_appcontext
    # def teardown_db(exception=None):
    #     db = g.pop('db_conn', None)
    #     if db is not None:
    #         close_db_connection(db) # Return to pool

    # --- Register Error Handlers ---
    register_error_handlers(app)

    # --- Simple Root Route ---
    @app.route('/')
    def index():
         return jsonify(message="Welcome to the Integrated Medical API", status="OK"), 200


    app.logger.info("Application initialization complete.")
    return app


def register_error_handlers(app):
    """Registers custom error handlers for the Flask app."""

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Handles HTTP exceptions (like BadRequest, NotFound, etc.)."""
        response = e.get_response()
        # Replace the body with JSON
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "error": e.description,
        })
        response.content_type = "application/json"
        app.logger.warning(f"HTTP Exception: {e.code} {e.name} - {e.description} ({request.path})")
        return response

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        """Handles any non-HTTP exception (like 500 Internal Server Error)."""
        # Log the full error traceback
        app.logger.error(f"Unhandled Exception: {e}", exc_info=True)

        # Return a generic JSON error message in production
        error_message = "An internal server error occurred."
        if app.debug:
            error_message = f"Unhandled Exception: {type(e).__name__} - {str(e)}"

        return jsonify(error=error_message, code=500, name="Internal Server Error"), 500

    app.logger.info("Custom error handlers registered.")