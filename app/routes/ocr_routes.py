# app/routes/ocr_routes.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, UnsupportedMediaType, InternalServerError
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
import os
import logging

from ..services.ocr_processing import process_image_ocr

ocr_bp = Blueprint('ocr_bp', __name__)
logger = logging.getLogger(__name__)

def _is_allowed_image_file(filename, mimetype):
    """Checks if the file extension and mimetype are allowed images."""
    allowed_ext = current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    allowed_mime = current_app.config['ALLOWED_IMAGE_MIMETYPES']
    _, ext = os.path.splitext(filename)
    # Check both extension and mimetype for robustness
    return ext.lower() in allowed_ext and mimetype in allowed_mime

@ocr_bp.route('/process-image', methods=['POST'])
def process_image_route():
    """Processes an uploaded image using OCR to extract relevant data."""
    log = current_app.logger
    log.info(f"Received request for /process-image from {request.remote_addr}")

    if 'image' not in request.files:
        log.warning("OCR Bad Request: No 'image' file part.")
        raise BadRequest("No image file provided.")

    image_file = request.files['image']
    disease_type = request.form.get('diseaseType')

    if not image_file.filename:
        log.warning("OCR Bad Request: Image file part has no filename.")
        raise BadRequest("Image file has no filename.")

    if not disease_type:
        log.warning("OCR Bad Request: 'diseaseType' form field missing.")
        raise BadRequest("Disease type is required.")

    allowed_disease_types = ['diabetes', 'heart_disease', 'parkinsons']
    if disease_type not in allowed_disease_types:
         log.warning(f"OCR Bad Request: Invalid disease type '{disease_type}'.")
         raise BadRequest(f"Invalid disease type. Allowed types: {', '.join(allowed_disease_types)}")

    if not _is_allowed_image_file(image_file.filename, image_file.mimetype):
        log.warning(f"OCR Unsupported Type: Received file '{image_file.filename}' type '{image_file.mimetype}'.")
        allowed_str = ", ".join(current_app.config['ALLOWED_IMAGE_EXTENSIONS'])
        raise UnsupportedMediaType(f"Unsupported image type. Allowed: {allowed_str}")

    # File size is checked by Flask's MAX_CONTENT_LENGTH

    upload_folder = current_app.config['OCR_UPLOAD_FOLDER']
    # Save the file temporarily - using a unique name within the uploads folder
    # Consider using tempfile.NamedTemporaryFile for automatic cleanup if preferred
    _, file_extension = os.path.splitext(image_file.filename)
    temp_filename = secure_filename(f"{disease_type}_{os.urandom(8).hex()}{file_extension}")
    temp_filepath = upload_folder / temp_filename
    saved = False

    try:
        log.debug(f"Saving uploaded image temporarily to: {temp_filepath}")
        image_file.save(str(temp_filepath))
        saved = True
        log.info(f"Saved image for OCR: {temp_filepath.name} ({temp_filepath.stat().st_size} bytes)")

        # --- Perform OCR Processing (Synchronous for now) ---
        # This could be slow; consider running in a thread for async behavior if needed
        extracted_text, extracted_data = process_image_ocr(temp_filepath, disease_type)

        if extracted_text is None and extracted_data is None:
            # This indicates an error during OCR processing itself
             log.error("OCR processing failed for the image.")
             raise InternalServerError("Failed to process image using OCR.")

        # Check if data extraction was successful based on the disease type logic
        if extracted_data is None:
             log.warning(f"OCR succeeded, but failed to extract structured data for type '{disease_type}'.")
             # Return success but indicate data extraction issues
             return jsonify({
                 "success": True, # OCR worked
                 "message": "OCR successful, but data extraction failed or was incomplete.",
                 "extractedText": extracted_text[:1000], # Return part of the text
                 "extractedData": {} # Return empty data
             })


        log.info(f"OCR processing successful for {disease_type}. Found {len([v for v in extracted_data.values() if v is not None])} fields.")
        return jsonify({
            "success": True,
            "extractedText": extracted_text[:1000], # Limit text size in response
            "extractedData": extracted_data # Send the structured data
        })

    except (BadRequest, UnsupportedMediaType) as e:
        raise e # Re-raise client errors
    except Exception as e:
         log.error(f"Unexpected error during OCR request processing: {e}", exc_info=True)
         raise InternalServerError("An unexpected error occurred during image processing.")
    finally:
        # --- Cleanup ---
        if saved and temp_filepath.exists():
            try:
                os.unlink(temp_filepath)
                log.debug(f"Deleted temporary OCR file: {temp_filepath}")
            except OSError as e:
                log.error(f"Error deleting temporary OCR file {temp_filepath}: {e}")