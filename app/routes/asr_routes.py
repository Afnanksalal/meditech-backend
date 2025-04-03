# api/app/routes/asr_routes.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, UnsupportedMediaType, InternalServerError
from werkzeug.utils import secure_filename
import tempfile
import librosa
import numpy as np
from pathlib import Path
import asyncio
import os
import logging

# Assuming these imports are correct relative to your project structure
from ..services.audio_processing import convert_audio
from ..services.gemini_api import translate_with_gemini, extract_emr, generate_suggestions
# Assuming detect_language_from_audio and run_pipeline_async are in transcription.py
from ..services.transcription import detect_language_from_audio, run_pipeline_async

asr_bp = Blueprint('asr_bp', __name__)
logger = logging.getLogger(__name__) # Use standard logger

# Allowed languages for explicit selection
ALLOWED_LANGUAGES = {'en', 'ml'}

def _is_allowed_audio_file(filename, mimetype):
    """Checks if the file extension and mimetype are allowed."""
    allowed_ext = current_app.config.get('ALLOWED_AUDIO_EXTENSIONS', set()) # Default to empty set
    allowed_mime = current_app.config.get('ALLOWED_AUDIO_MIMETYPES', set()) # Default to empty set
    _, ext = os.path.splitext(filename)
    # Check membership using 'in'
    return ext.lower() in allowed_ext and mimetype in allowed_mime

@asr_bp.route("/asr", methods=["POST"])
async def transcribe_audio_route():
    """
    Asynchronous endpoint for ASR, translation (conditional), EMR extraction, and suggestions.
    Allows optional language specification ('en' or 'ml') via form data to bypass detection.
    """
    logger.info(f"Received request for /api/asr from {request.remote_addr}")

    # --- Parameter Extraction and Validation ---
    if "audio" not in request.files:
        logger.warning("ASR Bad Request: No 'audio' file part.")
        raise BadRequest("No audio file provided in the 'audio' field.")

    audio_file = request.files["audio"]
    specified_language = request.form.get('language') # Optional form field for language

    if not audio_file.filename:
         logger.warning("ASR Bad Request: Audio file part has no filename.")
         raise BadRequest("Audio file has no filename.")

    # Validate audio file type
    if not _is_allowed_audio_file(audio_file.filename, audio_file.mimetype):
        logger.warning(f"ASR Unsupported Type: Received file '{audio_file.filename}' type '{audio_file.mimetype}'.")
        allowed_ext_str = ", ".join(current_app.config.get('ALLOWED_AUDIO_EXTENSIONS', []))
        allowed_mime_str = ", ".join(current_app.config.get('ALLOWED_AUDIO_MIMETYPES', []))
        raise UnsupportedMediaType(f"Unsupported audio type. Allowed extensions: {allowed_ext_str}. Allowed MIME types: {allowed_mime_str}")

    # Validate specified language (if provided)
    effective_language = None
    detection_method = "automatic" # Default
    if specified_language:
        specified_language = specified_language.lower().strip()
        if specified_language in ALLOWED_LANGUAGES:
            effective_language = specified_language
            detection_method = "specified"
            logger.info(f"User specified language: '{effective_language}'")
        else:
            logger.warning(f"ASR Bad Request: Invalid language '{specified_language}' specified. Must be one of {ALLOWED_LANGUAGES}.")
            raise BadRequest(f"Invalid language specified. Allowed values are: {', '.join(ALLOWED_LANGUAGES)}")

    # --- Processing ---
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        try:
            _, input_extension = os.path.splitext(audio_file.filename.lower())
            safe_filename = secure_filename(f"input_audio{input_extension}")
            input_path = tmp_dir / safe_filename
            logger.debug(f"Saving uploaded audio to temporary path: {input_path}")
            # Use asyncio.to_thread for blocking file I/O
            await asyncio.to_thread(audio_file.save, str(input_path))
            file_size = input_path.stat().st_size
            logger.info(f"Saved {file_size} bytes to {input_path}")

            # --- Step 1: Convert Audio ---
            output_path = tmp_dir / "processed_audio.wav"
            logger.info(f"Converting audio from '{input_path}' to '{output_path}'...")
            # Use asyncio.to_thread for blocking audio conversion process
            conversion_success = await asyncio.to_thread(convert_audio, input_path, output_path)
            if not conversion_success:
                 logger.error(f"Audio conversion failed for '{input_path}'.")
                 raise InternalServerError("Audio processing failed during conversion.")
            logger.info("Audio conversion completed.")

            # --- Step 2: Load Audio ---
            logger.info(f"Loading converted audio from '{output_path}'...")
            target_sr = current_app.config.get('TARGET_SAMPLE_RATE', 16000) # Default SR
            # Use asyncio.to_thread for blocking librosa loading
            audio_data, sr = await asyncio.to_thread(
                librosa.load, output_path, sr=target_sr
            )
            if audio_data is None or len(audio_data) == 0:
                 logger.error(f"Failed to load audio data from '{output_path}' or data is empty.")
                 # Use BadRequest as the input might be corrupted or silent
                 raise BadRequest("Invalid audio data: Could not load or data is empty after conversion.")
            if sr != target_sr:
                logger.warning(f"Loaded audio sample rate ({sr}Hz) differs from target ({target_sr}Hz). Librosa handled resampling.")
            logger.info(f"Loaded audio: {len(audio_data)} samples, {sr}Hz")
            # Prepare input dict for pipeline (ensure it matches pipeline expectations)
            audio_input_for_pipeline = {"raw": audio_data, "sampling_rate": sr}

            # --- Step 3: Determine Language (Detect or Use Specified) ---
            if effective_language is None: # Only detect if language wasn't specified
                logger.info("No language specified, proceeding with automatic detection...")
                detected_lang = await detect_language_from_audio(audio_data, sr)
                # Handle potential None or error return from detection (though current impl defaults to 'en')
                if not detected_lang or detected_lang not in ALLOWED_LANGUAGES:
                    logger.warning(f"Language detection returned invalid or unexpected result: '{detected_lang}'. Defaulting to English.")
                    effective_language = 'en'
                else:
                    effective_language = detected_lang
                logger.info(f"Automatically detected language: {effective_language}")
                detection_method = "automatic"
            # else: effective_language was already set from the valid 'specified_language'


            # --- Step 4: Conditional Transcription & Translation ---
            raw_transcription = ""
            final_english_text = "" # This will hold the text for EMR/Suggestions
            transcription_model_key = ""

            if effective_language == "ml":
                transcription_model_key = "whisper_ml"
                logger.info(f"Language is Malayalam ('{effective_language}'). Running '{transcription_model_key}' pipeline...")
                transcription_result = await run_pipeline_async(transcription_model_key, audio_input_for_pipeline)
                raw_transcription = transcription_result.get("text", "").strip() # .get handles key errors, strip handles whitespace

                # Check for pipeline errors indicated in the text itself
                if "Error:" in raw_transcription:
                     logger.error(f"Malayalam transcription pipeline reported an error: {raw_transcription}")
                     # We might want to return this error clearly to the user
                     raise InternalServerError(f"Transcription failed: {raw_transcription}")
                elif not raw_transcription:
                    logger.warning("Malayalam transcription result is empty.")
                    # No text to translate or process further
                else:
                    logger.info(f"Raw Malayalam Transcription ({len(raw_transcription)} chars): '{raw_transcription[:150]}...'")
                    logger.info("Translating Malayalam transcription to English...")
                    try:
                        # Assuming translate_with_gemini is async or wrapped appropriately
                        final_english_text = await translate_with_gemini(raw_transcription)
                        if not final_english_text or "unavailable" in final_english_text.lower():
                             logger.warning("Translation step failed or was unavailable. Proceeding without translated text.")
                             final_english_text = "" # Clear it if translation failed
                        else:
                            logger.info(f"Translated English Text ({len(final_english_text)} chars): '{final_english_text[:150]}...'")
                    except Exception as translation_err:
                         logger.error(f"Error during translation: {translation_err}", exc_info=True)
                         final_english_text = "" # Ensure it's empty on error

            else: # Default to English pipeline ('en' or if detection somehow failed)
                if effective_language != "en":
                    logger.warning(f"Effective language is '{effective_language}', but defaulting to English ('en') pipeline logic.")
                effective_language = 'en' # Ensure consistency
                transcription_model_key = "whisper_en"
                logger.info(f"Language is English ('{effective_language}'). Running '{transcription_model_key}' pipeline...")

                transcription_result = await run_pipeline_async(transcription_model_key, audio_input_for_pipeline)
                raw_transcription = transcription_result.get("text", "").strip()
                final_english_text = raw_transcription # English transcription IS the final English text

                if "Error:" in raw_transcription:
                     logger.error(f"English transcription pipeline reported an error: {raw_transcription}")
                     raise InternalServerError(f"Transcription failed: {raw_transcription}")
                elif not final_english_text:
                     logger.warning("English transcription result is empty.")
                else:
                     logger.info(f"English Transcription / Final Text ({len(final_english_text)} chars): '{final_english_text[:150]}...'")


            # --- Step 5: EMR Extraction and Suggestions (Using final_english_text) ---
            emr_data = {}
            suggestions = {}

            # Proceed only if we have valid, non-error final English text
            if final_english_text and "Error:" not in final_english_text:
                logger.info("Starting EMR extraction and suggestion generation using final English text...")
                try:
                    # Run EMR extraction and suggestions (potentially concurrently if desired)
                    # Use asyncio.gather if they can run in parallel and are async
                    # tasks = [
                    #     extract_emr(final_english_text),
                    #     generate_suggestions_based_on_text(final_english_text) # Hypothetical function if needed
                    # ]
                    # results = await asyncio.gather(*tasks, return_exceptions=True)
                    # emr_result = results[0]
                    # suggestions_result = results[1]

                    # Sequential execution for simplicity:
                    emr_data_task = extract_emr(final_english_text)
                    emr_data = await emr_data_task
                    logger.info(f"EMR Extraction Result: {emr_data}") # Log the raw result

                    # Check EMR data before generating suggestions based on it
                    if isinstance(emr_data, dict) and emr_data:
                         # Basic check if EMR data is meaningful (not just placeholders/errors)
                         is_meaningful_emr = any(
                             v and isinstance(v, str) and v.lower() not in ['not mentioned', 'none', 'n/a', '']
                             for v in emr_data.values()
                         )
                         if is_meaningful_emr:
                             logger.info("EMR data seems meaningful, generating suggestions based on it.")
                             suggestions_task = generate_suggestions(emr_data) # Pass extracted EMR
                             suggestions = await suggestions_task
                             logger.info(f"Suggestion Generation Result: {suggestions}")
                         else:
                              logger.warning("EMR data seems empty or non-informative; skipping suggestion generation based on EMR.")
                              suggestions = {"info": "Suggestions not generated due to non-informative EMR data."}
                    elif isinstance(emr_data, dict): # EMR returned an empty dict
                         logger.warning("EMR extraction returned an empty dictionary; skipping suggestion generation.")
                         suggestions = {"info": "Suggestions not generated because EMR extraction was empty."}
                    else: # EMR extraction failed or returned unexpected type
                         logger.warning(f"EMR extraction did not return a valid dictionary (type: {type(emr_data)}); skipping suggestion generation.")
                         # Ensure emr_data reflects the failure if it wasn't already an error dict
                         if not isinstance(emr_data, dict) or "error" not in emr_data:
                              emr_data = {"error": f"EMR extraction failed or returned invalid type: {type(emr_data).__name__}"}
                         suggestions = {"error": "Suggestions not generated due to EMR extraction failure."}

                except Exception as gemini_error:
                    logger.error(f"Error during Gemini EMR/Suggestion call: {gemini_error}", exc_info=True)
                    # Set defaults but don't crash the whole request, provide error info
                    emr_data = emr_data if isinstance(emr_data, dict) else {"error": "EMR processing failed"}
                    suggestions = {"error": f"Failed to process text with Gemini: {gemini_error}"}
            else:
                 logger.warning("Skipping EMR/Suggestion generation due to missing, empty, or failed transcription/translation.")
                 emr_data = {"info": "EMR not generated due to issues in prior steps."}
                 suggestions = {"info": "Suggestions not generated due to issues in prior steps."}


            # --- Step 6: Prepare and Send Response ---
            response_data = {
                "status": "success",
                "detection_method": detection_method, # 'specified' or 'automatic'
                "effective_language": effective_language, # The language used for transcription ('en' or 'ml')
                "raw_transcription": raw_transcription, # Direct output from the chosen Whisper model
                "processed_text": final_english_text, # The English text used for EMR/Suggestions (empty if ML->EN translation failed)
                "emr": emr_data,
                "suggestions": suggestions
            }
            logger.info("ASR Processing complete. Sending success response.")
            return jsonify(response_data)

        # --- Exception Handling ---
        except (BadRequest, UnsupportedMediaType) as e:
             # These are client errors, re-raise them for Flask to handle (usually 4xx response)
             logger.warning(f"Client Error during ASR: {type(e).__name__}({e.code}): {e.description}")
             raise e # Re-raise the original Werkzeug exception
        except FileNotFoundError as e:
             # Indicates a potential server setup issue (e.g., ffmpeg missing during conversion)
             logger.critical(f"ASR - Missing dependency or file? Error: {e}", exc_info=True)
             raise InternalServerError("Server configuration error: A required tool or file was not found.")
        except ValueError as e: # e.g., Librosa loading issues or other data processing errors
            logger.error(f"ASR - Data processing error: {e}", exc_info=True)
            # Treat as BadRequest as it might stem from invalid input data
            raise BadRequest(f"Invalid audio data or processing failed: {e}")
        except InternalServerError as e:
            # Re-raise specific internal server errors we've identified (like conversion/transcription failure)
            logger.error(f"ASR - Explicit Internal Server Error: {e.description}", exc_info=True)
            raise e
        except Exception as e:
            # Catch-all for unexpected errors during processing
            logger.error(f"Unexpected error during ASR processing: {e}", exc_info=True)
            error_msg = "An unexpected error occurred during audio processing."
            # Avoid leaking detailed internal errors unless in debug mode
            # if current_app.debug:
            #     error_msg += f" ({type(e).__name__}: {e})"
            raise InternalServerError(error_msg)
        # No finally needed, 'with tempfile.TemporaryDirectory()' handles cleanup