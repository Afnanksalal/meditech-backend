# app/services/transcription.py
import logging
import asyncio
import numpy as np
from flask import current_app
from langdetect import detect, detect_langs, DetectorFactory # Import langdetect
from langdetect.lang_detect_exception import LangDetectException # Import specific exception
from ..models import asr_models

logger = logging.getLogger(__name__)

# --- Ensure Deterministic Results ---
# Seed the DetectorFactory for consistent results (important!)
# Do this once when the module loads.
try:
    DetectorFactory.seed = 0
    logger.info("langdetect.DetectorFactory seeded for deterministic results.")
except NameError:
    logger.error("Failed to seed DetectorFactory. langdetect might not be installed correctly.")
except Exception as seed_err:
     logger.error(f"Error seeding DetectorFactory: {seed_err}", exc_info=True)


async def run_pipeline_async(model_key: str, audio_input: dict):
    """
    Runs a transformer pipeline model asynchronously in a thread pool executor.
    Handles different output types and ensures a consistent dictionary format.
    """
    loop = asyncio.get_running_loop()
    pipeline = asr_models.get(model_key)

    if not pipeline:
        logger.error(f"ASR pipeline '{model_key}' is not loaded or not found.")
        # Return error in the expected dictionary format
        return {"text": f"Error: Pipeline '{model_key}' not available."}

    logger.debug(f"Running pipeline '{model_key}' in executor thread...")
    try:
        # Important: Pass audio_input directly if pipeline expects it as a single arg
        # If pipeline expects kwargs, use pipeline(**audio_input)
        result = await loop.run_in_executor(
            None,
            pipeline,
            audio_input # Assuming pipeline takes the dict as the first argument
        )
        logger.debug(f"Pipeline '{model_key}' execution finished.")

        # --- Standardize Output ---
        text_output = ""
        if isinstance(result, dict) and "text" in result and isinstance(result["text"], str):
            text_output = result["text"].strip()
        elif isinstance(result, str):
            logger.debug(f"Pipeline '{model_key}' returned string directly. Wrapping in dict.")
            text_output = result.strip()
        else:
            logger.warning(f"Pipeline '{model_key}' returned unexpected type: {type(result)}. Attempting conversion.")
            try:
                # Attempt to convert non-dict/non-str results, defaulting to empty string
                text_output = str(result).strip() if result is not None else ""
            except Exception as conversion_err:
                logger.error(f"Could not convert pipeline result to string: {conversion_err}")
                text_output = "" # Ensure empty string on conversion failure

        return {"text": text_output}

    except Exception as e:
        logger.error(f"Error running pipeline '{model_key}': {e}", exc_info=True)
        # Return error in the expected dictionary format
        return {"text": f"Error during transcription processing with '{model_key}'."}


async def detect_language_from_audio(audio_data: np.ndarray, sr: int) -> str:
    """
    Detects the primary language (English or Malayalam) from an initial audio chunk
    by running both EN and ML ASR models concurrently and using the 'langdetect'
    library on their text outputs.

    Args:
        audio_data: The full audio data as a NumPy array.
        sr: The sample rate of the audio data.

    Returns:
        A string: "ml" for Malayalam, "en" for English. Defaults to "en".
    """
    logger.info(f"Starting automatic language detection using langdetect from audio sample (shape: {audio_data.shape}, sr: {sr})")
    chunk_size_ms = current_app.config.get('CHUNK_SIZE_MS', 5000) # Default 5s
    min_text_len_for_detect = current_app.config.get('MIN_TEXT_LEN_FOR_DETECT', 10) # Configurable min length

    try:
        if not isinstance(audio_data, np.ndarray) or audio_data.ndim != 1 or audio_data.size == 0:
             logger.error(f"Invalid audio_data provided for language detection: type {type(audio_data)}, shape {getattr(audio_data, 'shape', 'N/A')}")
             return "en" # Default on invalid input

        if sr <= 0:
            logger.error(f"Invalid sample rate for language detection: {sr}")
            return "en" # Default on invalid sample rate

        chunk_size_samples = int(chunk_size_ms / 1000 * sr)
        if chunk_size_samples <= 0:
             logger.warning(f"Calculated chunk size is zero or negative ({chunk_size_samples} samples). Using full audio for detection.")
             chunk = audio_data
        elif chunk_size_samples >= audio_data.size:
             logger.debug(f"Chunk size ({chunk_size_samples}) >= audio length ({audio_data.size}). Using full audio for detection.")
             chunk = audio_data
        else:
            chunk = audio_data[:chunk_size_samples]
            logger.debug(f"Using initial chunk of {chunk.size} samples for detection.")


        if chunk is None or len(chunk) == 0:
             logger.warning("Cannot detect language from empty or invalid audio chunk.")
             return "en"

        # Ensure models needed for detection are loaded
        ml_model_key = 'whisper_ml'
        en_model_key = 'whisper_en'
        if not asr_models.get(ml_model_key) or not asr_models.get(en_model_key):
             logger.error(f"One or both ASR models ('{ml_model_key}', '{en_model_key}') needed for language detection are not loaded.")
             return "en" # Cannot perform detection

        audio_input = {"raw": chunk, "sampling_rate": sr}

        logger.info(f"Running {ml_model_key} & {en_model_key} models concurrently on audio chunk for langdetect analysis...")
        ml_task = run_pipeline_async(ml_model_key, audio_input)
        en_task = run_pipeline_async(en_model_key, audio_input)

        # Use return_exceptions=True to handle potential errors in one pipeline without stopping the other
        results = await asyncio.gather(ml_task, en_task, return_exceptions=True)
        ml_result, en_result = results[0], results[1]

        # --- Extract text, robustly handling errors and dict format ---
        ml_text = ""
        if isinstance(ml_result, dict) and "text" in ml_result and isinstance(ml_result.get("text"), str):
            ml_text = ml_result["text"] # Already stripped in run_pipeline_async
            if "Error:" not in ml_text:
                 logger.debug(f"Lang Detect - ML Model Output ({len(ml_text)} chars): '{ml_text[:100]}...'")
            else:
                 logger.warning(f"Lang Detect - ML Model reported error: {ml_text}")
                 ml_text = "" # Treat errors as empty text for detection purposes
        elif isinstance(ml_result, Exception):
            logger.error(f"Error during ML model inference for lang detect: {ml_result}", exc_info=ml_result)
        else:
            logger.warning(f"Non-dict or unexpected result from ML model for lang detect: {ml_result}")

        en_text = ""
        if isinstance(en_result, dict) and "text" in en_result and isinstance(en_result.get("text"), str):
            en_text = en_result["text"] # Already stripped
            if "Error:" not in en_text:
                logger.debug(f"Lang Detect - EN Model Output ({len(en_text)} chars): '{en_text[:100]}...'")
            else:
                logger.warning(f"Lang Detect - EN Model reported error: {en_text}")
                en_text = "" # Treat errors as empty text
        elif isinstance(en_result, Exception):
             logger.error(f"Error during EN model inference for lang detect: {en_result}", exc_info=en_result)
        else:
             logger.warning(f"Non-dict or unexpected result from EN model for lang detect: {en_result}")

        # --- Language Detection using langdetect ---
        detected_lang_ml = None
        detected_lang_en = None

        # Try detecting language on ML output if long enough
        if ml_text and len(ml_text) >= min_text_len_for_detect:
            try:
                # Use detect_langs to potentially get probabilities later if needed
                langs = detect_langs(ml_text)
                if langs:
                    detected_lang_ml = langs[0].lang # Get the top language
                    logger.info(f"langdetect result on ML output ('{ml_text[:50]}...'): {langs}")
                else:
                     logger.warning(f"langdetect returned empty result for ML output: '{ml_text[:50]}...'")
            except LangDetectException:
                logger.warning(f"langdetect failed on ML output (likely too short or ambiguous): '{ml_text[:50]}...'")
            except Exception as e:
                 logger.error(f"Unexpected error during langdetect on ML output: {e}", exc_info=True)

        # Try detecting language on EN output if long enough
        if en_text and len(en_text) >= min_text_len_for_detect:
             try:
                 langs = detect_langs(en_text)
                 if langs:
                     detected_lang_en = langs[0].lang # Get the top language
                     logger.info(f"langdetect result on EN output ('{en_text[:50]}...'): {langs}")
                 else:
                    logger.warning(f"langdetect returned empty result for EN output: '{en_text[:50]}...'")
             except LangDetectException:
                 logger.warning(f"langdetect failed on EN output (likely too short or ambiguous): '{en_text[:50]}...'")
             except Exception as e:
                 logger.error(f"Unexpected error during langdetect on EN output: {e}", exc_info=True)


        # --- Decision Logic based on langdetect results ---
        # This logic prioritizes cases where the detected language matches the model's intended language.

        # Case 1: ML model output looks like ML, EN model output doesn't look like EN
        if detected_lang_ml == 'ml' and detected_lang_en != 'en':
            logger.info("Decision: Classifying as Malayalam (ML output detected 'ml', EN output did not detect 'en').")
            return "ml"

        # Case 2: EN model output looks like EN, ML model output doesn't look like ML
        if detected_lang_en == 'en' and detected_lang_ml != 'ml':
             logger.info("Decision: Classifying as English (EN output detected 'en', ML output did not detect 'ml').")
             return "en"

        # Case 3: Both models' outputs are detected as ML
        if detected_lang_ml == 'ml' and detected_lang_en == 'ml':
            logger.info("Decision: Classifying as Malayalam (Both models' outputs detected as 'ml').")
            return "ml"

        # Case 4: Both models' outputs are detected as EN
        if detected_lang_ml == 'en' and detected_lang_en == 'en':
            logger.info("Decision: Classifying as English (Both models' outputs detected as 'en').")
            return "en"

        # Case 5: Ambiguous or Conflicting - ML output detected 'ml', EN inconclusive
        if detected_lang_ml == 'ml' and detected_lang_en is None:
             logger.info("Decision: Classifying as Malayalam (ML output detected 'ml', EN detection inconclusive).")
             return "ml"

        # Case 6: Ambiguous or Conflicting - EN output detected 'en', ML inconclusive
        if detected_lang_en == 'en' and detected_lang_ml is None:
             logger.info("Decision: Classifying as English (EN output detected 'en', ML detection inconclusive).")
             return "en"

        # Case 7: All other scenarios (including both failed detection, or conflicts like ml->en and en->ml)
        # Defaulting to English is a common fallback. Could potentially use text length as a weak heuristic here,
        # but defaulting is simpler.
        logger.warning(f"Language detection ambiguous or conflicting (ml_detect='{detected_lang_ml}', en_detect='{detected_lang_en}'). Defaulting to English.")
        return "en"

    except Exception as e:
        logger.error(f"Automatic language detection failed unexpectedly: {e}", exc_info=True)
        return "en" # Fallback to English on any unexpected error during detection