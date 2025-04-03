# app/services/ocr_processing.py
import pytesseract
from PIL import Image # Pillow for image handling
import logging
import re
from pathlib import Path
from flask import current_app

logger = logging.getLogger(__name__)

# --- OCR Text Extraction Helpers ---

def _extract_value(text: str, labels: list[str]) -> str | None:
    """Finds the first value associated with any of the labels."""
    # Try to find patterns like "Label: 123.45" or "Label 123.45"
    # Handles integers, decimals, possibly ranges separated by '-'
    # Makes labels case-insensitive in the regex
    for label in labels:
        # Regex explanation:
        # (?i)             : Case-insensitive flag
        # label            : The literal label text (regex-escaped)
        # [\s:]+           : One or more whitespace characters or colons
        # (                : Start capturing group for the value
        #   [\-\+]?        : Optional sign (+ or -)
        #   \d+            : One or more digits
        #   (?:            : Start non-capturing group for optional decimal part
        #     \.           : Literal dot
        #     \d+          : One or more digits
        #   )?             : End optional non-capturing group (decimal part)
        #   (?:            : Start non-capturing group for optional range
        #     \s*-\s*      : Hyphen surrounded by optional whitespace
        #     [\-\+]?\d+(?:\.\d+)? : Another number (integer or decimal)
        #   )?             : End optional non-capturing group (range)
        # )                : End capturing group for the value
        pattern = rf"(?i){re.escape(label)}[\s:]+([\-\+]?\d+(?:\.\d+)?(?:\s*-\s*[\-\+]?\d+(?:\.\d+)?)?)"
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            logger.debug(f"Extracted value '{value}' for label '{label}'")
            # Basic cleanup (remove potential units sometimes captured if regex is too broad)
            # This needs refinement based on expected units
            value = re.sub(r'\s*(mg/dl|mmhg|kg/m2).*$', '', value, flags=re.IGNORECASE).strip()
            return value
    return None


def _extract_data_from_text(text: str, field_map: dict[str, list[str]]) -> dict:
    """Generic function to extract data based on field mapping."""
    results = {}
    fields_found_count = 0
    lower_text = text.lower() # Process lower text for potentially better matching robustness

    for field_id, labels in field_map.items():
        value = _extract_value(text, labels) # Use original text for extraction accuracy with regex case insensitivity
        if value is not None: # Check for None explicitly, as "0" is a valid value
            results[field_id] = value
            fields_found_count += 1
        else:
             results[field_id] = None # Explicitly mark as not found

    logger.info(f"OCR Extraction: Found {fields_found_count}/{len(field_map)} fields.")
    return results


# --- Disease-Specific Processing Logic ---

def _process_diabetes_text(text: str) -> dict:
    field_map = {
        'Pregnancies': ['pregnancies', 'pregnancy', 'number of pregnancies'],
        'Glucose': ['glucose', 'blood glucose', 'glucose level', 'sugar level'],
        'BloodPressure': ['blood pressure', 'bp', 'systolic pressure'],
        'SkinThickness': ['skin thickness', 'skin fold thickness', 'triceps skinfold'],
        'Insulin': ['insulin', 'serum insulin', 'insulin level'],
        'BMI': ['bmi', 'body mass index'],
        'DiabetesPedigreeFunction': ['diabetes pedigree', 'dpf', 'pedigree function'],
        'Age': ['age', 'patient age', 'years old']
    }
    return _extract_data_from_text(text, field_map)

def _process_heart_disease_text(text: str) -> dict:
    field_map = {
        'age': ['age', 'patient age', 'years old'],
        'sex': ['sex', 'gender'], # Needs special handling (0/1 vs M/F) post-extraction
        'cp': ['chest pain type', 'cp', 'chest pain'],
        'trestbps': ['resting blood pressure', 'trestbps', 'bp'],
        'chol': ['cholesterol', 'chol', 'serum cholestoral'],
        'fbs': ['fasting blood sugar > 120', 'fasting blood sugar', 'fbs'], # Needs special handling (0/1)
        'restecg': ['resting ecg', 'restecg', 'resting electrocardiographic'],
        'thalach': ['maximum heart rate achieved', 'max heart rate', 'thalach'],
        'exang': ['exercise induced angina', 'exang', 'exercise angina'], # Needs special handling (0/1)
        'oldpeak': ['st depression induced by exercise', 'oldpeak', 'st depression'],
        'slope': ['slope of the peak exercise st segment', 'slope', 'peak exercise slope'],
        'ca': ['number of major vessels colored by flourosopy', 'ca', 'major vessels'],
        'thal': ['thal', 'thalassemia'] # Needs special handling (mapping values)
    }
    extracted = _extract_data_from_text(text, field_map)

    # --- Post-processing for specific heart disease fields ---
    # Sex: Try to map M/F or Male/Female to 1/0 (assuming 1=Male, 0=Female per model training)
    if extracted.get('sex') is not None:
        sex_val = str(extracted['sex']).lower()
        if 'male' in sex_val or sex_val == 'm' or sex_val == '1':
            extracted['sex'] = '1'
        elif 'female' in sex_val or sex_val == 'f' or sex_val == '0':
            extracted['sex'] = '0'
        else:
             logger.warning(f"Could not map OCR 'sex' value '{extracted['sex']}' to 0 or 1.")
             extracted['sex'] = None # Or keep original / handle error

    # FBS: Check for presence of "> 120" or keywords indicating true/false
    if extracted.get('fbs') is not None:
        fbs_val = str(extracted['fbs']).lower()
        if 'true' in fbs_val or '> 120' in fbs_val or fbs_val == '1' or 'yes' in fbs_val:
             extracted['fbs'] = '1'
        elif 'false' in fbs_val or '<= 120' in fbs_val or fbs_val == '0' or 'no' in fbs_val:
             extracted['fbs'] = '0'
        else:
             # If just a number, assume it's the value, but model needs 0/1
             try:
                 numeric_fbs = float(fbs_val)
                 extracted['fbs'] = '1' if numeric_fbs > 120 else '0'
             except ValueError:
                  logger.warning(f"Could not map OCR 'fbs' value '{extracted['fbs']}' to 0 or 1.")
                  extracted['fbs'] = None

    # Exang: Map Yes/No or True/False to 1/0
    if extracted.get('exang') is not None:
         exang_val = str(extracted['exang']).lower()
         if 'yes' in exang_val or 'true' in exang_val or exang_val == '1':
              extracted['exang'] = '1'
         elif 'no' in exang_val or 'false' in exang_val or exang_val == '0':
              extracted['exang'] = '0'
         else:
              logger.warning(f"Could not map OCR 'exang' value '{extracted['exang']}' to 0 or 1.")
              extracted['exang'] = None

    # Thal: Map common values if possible (depends heavily on model training data)
    # Example mapping: 3 = normal; 6 = fixed defect; 7 = reversable defect --> Model likely expects numbers 1,2,3 or similar
    # This requires knowing the exact mapping used for training the heart_disease_model.sav
    # For now, we'll just try to return the number found, or None.
    if extracted.get('thal') is not None:
         thal_val = str(extracted['thal'])
         match = re.search(r'\d', thal_val) # Find the first digit
         if match:
              extracted['thal'] = match.group(0) # Keep only the number found
              # Add specific mapping here if known, e.g.:
              # if extracted['thal'] == '3': extracted['thal'] = '1' # map 'normal' to model's code '1'
         else:
              logger.warning(f"Could not extract numerical Thal value from '{extracted['thal']}'")
              extracted['thal'] = None

    return extracted


def _process_parkinsons_text(text: str) -> dict:
    # Parkinson's fields often have specific names, less variation needed?
    field_map = {
        'fo': ['mdvp:fo(hz)', 'average vocal fundamental frequency', 'fo(hz)'],
        'fhi': ['mdvp:fhi(hz)', 'maximum vocal fundamental frequency', 'fhi(hz)'],
        'flo': ['mdvp:flo(hz)', 'minimum vocal fundamental frequency', 'flo(hz)'],
        'Jitter_percent': ['mdvp:jitter(%)', 'jitter percent', 'jitter(%)'],
        'Jitter_Abs': ['mdvp:jitter(abs)', 'jitter absolute', 'jitter(abs)'],
        'RAP': ['mdvp:rap', 'rap'],
        'PPQ': ['mdvp:ppq', 'ppq'],
        'DDP': ['jitter:ddp', 'ddp'],
        'Shimmer': ['mdvp:shimmer', 'shimmer'],
        'Shimmer_dB': ['mdvp:shimmer(db)', 'shimmer(db)'],
        'APQ3': ['shimmer:apq3', 'apq3'],
        'APQ5': ['shimmer:apq5', 'apq5'],
        'APQ': ['mdvp:apq', 'apq'],
        'DDA': ['shimmer:dda', 'dda'],
        'NHR': ['nhr', 'noise-to-harmonic ratio'],
        'HNR': ['hnr', 'harmonic-to-noise ratio'],
        'RPDE': ['rpde', 'recurrence period density entropy'],
        'DFA': ['dfa', 'detrended fluctuation analysis'],
        'spread1': ['spread1'],
        'spread2': ['spread2'],
        'D2': ['d2', 'correlation dimension'],
        'PPE': ['ppe', 'pitch period entropy']
    }
    # Simple extraction might work better here if labels are consistent
    return _extract_data_from_text(text, field_map)


# --- Main OCR Service Function ---

def process_image_ocr(image_path: Path, disease_type: str) -> tuple[str | None, dict | None]:
    """
    Performs OCR on an image and extracts data based on disease type.

    Args:
        image_path: Path to the image file.
        disease_type: 'diabetes', 'heart_disease', or 'parkinsons'.

    Returns:
        A tuple containing:
        - The extracted text (str) or None if OCR failed.
        - A dictionary of extracted field values (dict) or None if processing failed.
    """
    try:
        # Configure Tesseract path if specified in config
        tesseract_cmd = current_app.config.get('TESSERACT_CMD')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        logger.info(f"Starting OCR for {disease_type} image: {image_path}")

        # Perform OCR - use appropriate config options if needed
        # '--psm 6' assumes a single uniform block of text. Adjust if needed.
        # '-l eng' specifies English language.
        custom_config = r'--oem 3 --psm 6' # Example config
        extracted_text = pytesseract.image_to_string(Image.open(image_path), lang='eng', config=custom_config)

        if not extracted_text:
            logger.warning(f"OCR returned empty text for image: {image_path}")
            return None, None # Indicate OCR failure

        logger.info(f"OCR extracted text (first 200 chars): {extracted_text[:200]}...")
        # logger.debug(f"Full OCR Text:\n{extracted_text}") # Optionally log full text in debug

        # Process based on disease type
        extracted_data = None
        if disease_type == 'diabetes':
            extracted_data = _process_diabetes_text(extracted_text)
        elif disease_type == 'heart_disease':
            extracted_data = _process_heart_disease_text(extracted_text)
        elif disease_type == 'parkinsons':
            extracted_data = _process_parkinsons_text(extracted_text)
        else:
            logger.error(f"Invalid disease type '{disease_type}' for OCR processing.")
            return extracted_text, None # Return text but indicate data processing failure

        # Return raw text and processed data
        return extracted_text, extracted_data

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract executable not found. Ensure it's installed and in PATH, or TESSERACT_CMD is set correctly in .env.")
        return None, None
    except Exception as e:
        logger.error(f"Error during OCR processing for {image_path}: {e}", exc_info=True)
        return None, None