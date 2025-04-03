# api/app/routes/predict_routes.py
from flask import Blueprint, request, jsonify, current_app
# Import the specific exception class
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import logging
import numpy as np

from ..models import prediction_models # Import loaded models
from ..database import db_create_room, db_check_room_exists, generate_room_code
from .. import active_rooms # Import shared active_rooms dict

predict_bp = Blueprint('predict_bp', __name__)
logger = logging.getLogger(__name__)


@predict_bp.route('/create_room', methods=['POST'])
def create_room_route():
    """Creates a new unique room and stores it."""
    max_retries = 5
    for _ in range(max_retries):
        room_id = generate_room_code()
        if room_id in active_rooms: continue # Check memory first
        if db_create_room(room_id): # Try DB
            active_rooms[room_id] = {'users': {}} # Add to memory on success
            logger.info(f"Room '{room_id}' created successfully (DB and memory).")
            return jsonify({'room_id': room_id}), 201
        logger.warning(f"DB creation failed for room code '{room_id}', retrying...")
    logger.error("Failed to create a unique room code after multiple retries.")
    raise InternalServerError('Failed to create room, please try again.')


@predict_bp.route('/check_room/<string:room_id>', methods=['GET'])
def check_room_route(room_id):
    """Checks if a room exists in the database."""
    if not room_id or len(room_id) != 6:
        raise BadRequest("Invalid room ID format.")
    exists = db_check_room_exists(room_id)
    return jsonify({'exists': exists})


@predict_bp.route('/predict_diabetes', methods=['POST'])
def predict_diabetes_route():
    """Predicts diabetes based on input features."""
    model = prediction_models.get('diabetes')
    if not model:
        logger.error("Diabetes prediction model is not loaded.")
        raise InternalServerError("Prediction service unavailable.")

    if not request.is_json:
        raise BadRequest("Request must be JSON.")

    data = request.json
    expected_keys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    try:
        input_values = []
        missing_keys = []
        for key in expected_keys:
            if key not in data:
                missing_keys.append(key)
            else:
                 try:
                     input_values.append(float(data[key]))
                 except (ValueError, TypeError):
                      raise BadRequest(f"Invalid non-numeric value provided for '{key}'.")

        if missing_keys:
            raise BadRequest(f"Missing required fields: {', '.join(missing_keys)}")

        input_array = np.array(input_values).reshape(1, -1)
        logger.debug(f"Diabetes input array: {input_array}")

        # Get the prediction (0 or 1)
        prediction = model.predict(input_array)

        # Determine the label based on the prediction
        result_label = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        # Log the result (without confidence)
        logger.info(f"Diabetes Prediction: {result_label}")

        # Return the prediction label and boolean flag
        return jsonify({
            'prediction': result_label,
            'is_diabetic': bool(prediction[0] == 1)
            })

    except BadRequest as e:
         raise e
    except Exception as e:
        logger.error(f"Error during diabetes prediction: {e}", exc_info=True)
        raise InternalServerError("Prediction failed.")


@predict_bp.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease_route():
    """Predicts heart disease based on input features."""
    model = prediction_models.get('heart_disease')
    if not model:
        logger.error("Heart disease prediction model is not loaded.")
        raise InternalServerError("Prediction service unavailable.")

    if not request.is_json:
        raise BadRequest("Request must be JSON.")

    data = request.json
    expected_keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    try:
        input_values = []
        missing_keys = []
        for key in expected_keys:
            if key not in data:
                missing_keys.append(key)
            else:
                 try:
                      input_values.append(float(data[key]))
                 except (ValueError, TypeError):
                      raise BadRequest(f"Invalid non-numeric value provided for '{key}'.")
        if missing_keys:
            raise BadRequest(f"Missing required fields: {', '.join(missing_keys)}")

        input_array = np.array(input_values).reshape(1, -1)
        logger.debug(f"Heart Disease input array: {input_array}")

        # Get the prediction (0 or 1)
        prediction = model.predict(input_array)

        # Determine the label based on the prediction
        result_label = 'Heart Disease Present' if prediction[0] == 1 else 'No Heart Disease'

        # Log the result (without confidence)
        logger.info(f"Heart Disease Prediction: {result_label}")

        # Return the prediction label and boolean flag
        return jsonify({
             'prediction': result_label,
             'has_heart_disease': bool(prediction[0] == 1)
             })

    except BadRequest as e:
         raise e
    except Exception as e:
        logger.error(f"Error during heart disease prediction: {e}", exc_info=True)
        raise InternalServerError("Prediction failed.")


@predict_bp.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons_route():
    """Predicts Parkinson's disease based on input features."""
    model = prediction_models.get('parkinsons')
    if not model:
        logger.error("Parkinson's prediction model is not loaded.")
        raise InternalServerError("Prediction service unavailable.")

    if not request.is_json:
        raise BadRequest("Request must be JSON.")

    data = request.json
    expected_keys = [
        'fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP',
        'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    try:
        input_values = []
        missing_keys = []
        for key in expected_keys:
            if key not in data:
                missing_keys.append(key)
            else:
                 try:
                      input_values.append(float(data[key]))
                 except (ValueError, TypeError):
                      raise BadRequest(f"Invalid non-numeric value provided for '{key}'.")
        if missing_keys:
            raise BadRequest(f"Missing required fields: {', '.join(missing_keys)}")

        input_array = np.array(input_values).reshape(1, -1)
        logger.debug(f"Parkinson's input array shape: {input_array.shape}")

        # Get the prediction (0 or 1)
        prediction = model.predict(input_array)

        # Determine the label based on the prediction
        result_label = "Parkinson's Disease Likely" if prediction[0] == 1 else "No Parkinson's Disease Likely"

        # Log the result (without confidence)
        logger.info(f"Parkinson's Prediction: {result_label}")

        # Return the prediction label and boolean flag
        return jsonify({
             'prediction': result_label,
             'has_parkinsons': bool(prediction[0] == 1)
             })

    except BadRequest as e:
         raise e
    except Exception as e:
        logger.error(f"Error during Parkinson's prediction: {e}", exc_info=True)
        raise InternalServerError("Prediction failed.")