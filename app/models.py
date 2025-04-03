# app/models.py
import pickle
import logging
from pathlib import Path
import torch
from transformers import pipeline # Keep pipeline import
# Removed AutoModelForSpeechSeq2Seq, AutoProcessor unless specifically needed elsewhere
from flask import current_app

logger = logging.getLogger(__name__)

# Global dictionaries to hold loaded models
prediction_models = {
    'diabetes': None,
    'heart_disease': None,
    'parkinsons': None
}

asr_models = {
    'whisper_en': None,
    'whisper_ml': None
}

def load_pickle_models():
    """Loads the scikit-learn prediction models from pickle files."""
    model_dir = current_app.config['PICKLE_MODELS_DIR']
    models_to_load = {
        'diabetes': 'diabetes_model.sav',
        'heart_disease': 'heart_disease_model.sav',
        'parkinsons': 'parkinsons_model.sav'
    }
    logger.info(f"Attempting to load prediction models from: {model_dir}")
    loaded_count = 0
    for key, filename in models_to_load.items():
        model_path = model_dir / filename
        try:
            if not model_path.exists():
                 logger.error(f"Pickle model file not found: {model_path}")
                 continue

            with open(model_path, 'rb') as f:
                prediction_models[key] = pickle.load(f)
            logger.info(f"Successfully loaded prediction model '{key}' from {model_path}")
            loaded_count += 1
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            # Log the specific error
            logger.error(f"Failed to load prediction model '{key}' from {model_path}: {e}", exc_info=True)
            # Optionally print scikit-learn version incompatibility warning here if detected in exception details

    if loaded_count != len(models_to_load):
        logger.warning("Not all prediction models were loaded successfully.")
    else:
        logger.info("All prediction models loaded.")

    return prediction_models # Return the dictionary


def load_hf_models():
    """Loads the Hugging Face ASR models."""
    # hf_cache_dir is determined by HF_HOME env var or default ~/.cache/huggingface/hub
    # We don't need to pass it to the pipeline directly.
    # hf_cache_dir = current_app.config['HF_MODELS_DIR'] # Removed usage

    try:
        logger.info("Initializing Hugging Face ASR models...")
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
            logger.info("CUDA available. Using GPU with float16.")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            logger.info("CUDA not available. Using CPU with float32.")


        # --- English Model ---
        en_model_id = "openai/whisper-small"
        logger.info(f"Loading English ASR model: {en_model_id} on device: {device}")
        try:
             asr_models['whisper_en'] = pipeline(
                 "automatic-speech-recognition",
                 model=en_model_id,
                 torch_dtype=torch_dtype,
                 device=device,
                 # Removed cache_dir argument
                 # model_kwargs={"attn_implementation": "flash_attention_2"} # Optional, requires specific setup
             )
             logger.info(f"English ASR model '{en_model_id}' pipeline created (download/load may happen on first use).")
             # Optional: Trigger download/load now with dummy data
             # logger.info("Performing dummy inference to ensure English model is loaded...")
             # _ = asr_models['whisper_en'](np.zeros(1000, dtype=np.float32)) # Requires numpy import
             # logger.info("Dummy inference for English model complete.")

        except Exception as e:
             logger.error(f"Failed to initialize English ASR pipeline '{en_model_id}': {e}", exc_info=True)


        # --- Malayalam Model ---
        ml_model_id = "kavyamanohar/whisper-small-malayalam"
        logger.info(f"Loading Malayalam ASR model: {ml_model_id} on device: {device}")
        try:
             asr_models['whisper_ml'] = pipeline(
                 "automatic-speech-recognition",
                 model=ml_model_id,
                 torch_dtype=torch_dtype,
                 device=device,
                 # Removed cache_dir argument
                 # model_kwargs={"attn_implementation": "flash_attention_2"} # Optional
             )
             logger.info(f"Malayalam ASR model '{ml_model_id}' pipeline created (download/load may happen on first use).")
             # Optional: Trigger download/load now
             # logger.info("Performing dummy inference to ensure Malayalam model is loaded...")
             # _ = asr_models['whisper_ml'](np.zeros(1000, dtype=np.float32)) # Requires numpy import
             # logger.info("Dummy inference for Malayalam model complete.")

        except Exception as e:
              logger.error(f"Failed to initialize Malayalam ASR pipeline '{ml_model_id}': {e}", exc_info=True)


        # Check if pipelines were created successfully (they might still fail on first use if download fails)
        if not asr_models.get('whisper_en') or not asr_models.get('whisper_ml'):
            logger.warning("One or both ASR model pipelines could not be initialized.")
        else:
            logger.info("ASR model pipelines initialized successfully.")

    except Exception as e:
        logger.error(f"General error during Hugging Face model loading setup: {e}", exc_info=True)

    return asr_models # Return the dictionary

