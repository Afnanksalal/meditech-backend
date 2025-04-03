# api/app/routes/suggestions_route.py
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, InternalServerError
import asyncio
import logging

# Assuming gemini_api.py is in the services directory relative to this file's parent
from ..services.gemini_api import call_gemini_api

suggestions_bp = Blueprint('suggestions_bp', __name__)
logger = logging.getLogger(__name__) # Use standard Flask logger

@suggestions_bp.route("/generate_doctor_suggestion", methods=["POST"])
async def generate_doctor_suggestion_route():
    """
    Asynchronous endpoint to generate a doctor specialty suggestion based on input.
    Expects JSON body: {"prediction": "...", "symptoms": "...", "health_records": "..."}
    Returns JSON: {"suggestion": "Specialty Name"} or {"error": "..."}
    """
    logger.info(f"Received request for /api/generate_doctor_suggestion from {request.remote_addr}")

    try:
        # Get JSON data from the request
        # Use force=True cautiously or add Content-Type check
        data = request.get_json()
        if not data:
            logger.warning("Doctor Suggestion - Bad Request: No JSON data received.")
            raise BadRequest("No JSON data received in the request body.")

        # Extract and validate inputs
        prediction = data.get('prediction', '').strip()
        symptoms = data.get('symptoms', '').strip()
        health_records = data.get('health_records', '').strip() # Match JS key

        # Basic validation: At least one field should have some content
        if not prediction and not symptoms and not health_records:
            logger.warning("Doctor Suggestion - Bad Request: All input fields (prediction, symptoms, health_records) are empty.")
            raise BadRequest("Please provide prediction result, symptoms, or health records.")

        logger.debug(f"Received doctor suggestion inputs - Prediction: '{prediction[:50]}...', Symptoms: '{symptoms[:50]}...', Records: '{health_records[:50]}...'")

        # --- Construct Prompt for Gemini ---
        # Instruct Gemini to return ONLY the specialty.
        prompt = f"""
        Based on the following patient information, recommend the single most appropriate medical specialty to consult.
        Return *only* the name of the medical specialty (e.g., Cardiologist, Neurologist, Endocrinologist, General Practitioner).
        Do not include any explanations, introductory phrases, or extra text.

        Patient Information:
        - Disease Prediction Result: {prediction if prediction else 'Not provided'}
        - Reported Symptoms: {symptoms if symptoms else 'Not provided'}
        - Relevant Health Records: {health_records if health_records else 'Not provided'}

        Recommended Specialty:
        """

        # --- Call Gemini API ---
        logger.info("Calling Gemini API for doctor specialty suggestion...")
        # Assuming call_gemini_api handles retries and internal errors
        suggestion_text = await call_gemini_api(prompt)

        # --- Process Result ---
        if suggestion_text:
            # Clean up potential extra words if Gemini doesn't follow instructions perfectly
            # (e.g., if it returns "Recommended Specialty: Cardiologist")
            suggestion = suggestion_text.split(':')[-1].strip() # Take text after last colon
             # Further simple cleaning
            suggestion = suggestion.replace('.', '').strip() # Remove trailing periods

            logger.info(f"Gemini suggested specialty: '{suggestion}'")
            return jsonify({"suggestion": suggestion}), 200
        else:
            logger.error("Gemini API call for doctor suggestion failed or returned no content.")
            # Let the user know the suggestion couldn't be generated
            raise InternalServerError("Could not generate doctor suggestion at this time.")

    except BadRequest as e:
        # Re-raise client errors for Flask to handle (400 response)
        logger.warning(f"Doctor Suggestion - Bad Request ({e.code}): {e.description}")
        raise e
    except Exception as e:
        # Catch-all for unexpected errors (Gemini API issues, etc.)
        logger.error(f"Unexpected error during doctor suggestion generation: {e}", exc_info=True)
        raise InternalServerError("An unexpected error occurred while generating the doctor suggestion.")


@suggestions_bp.route("/generate_diet_plan", methods=["POST"])
async def generate_diet_plan_route():
    """
    Asynchronous endpoint to generate a diet plan suggestion based on input.
    Expects JSON body: {"preferences": "...", "goals": "..."}
    Returns JSON: {"diet_plan": "Plan text..."} or {"error": "..."}
    """
    logger.info(f"Received request for /api/generate_diet_plan from {request.remote_addr}")

    try:
        data = request.get_json()
        if not data:
            logger.warning("Diet Plan - Bad Request: No JSON data received.")
            raise BadRequest("No JSON data received in the request body.")

        # Extract and validate inputs
        preferences = data.get('preferences', '').strip()
        goals = data.get('goals', '').strip()

        if not preferences and not goals:
            logger.warning("Diet Plan - Bad Request: Both 'preferences' and 'goals' fields are empty.")
            raise BadRequest("Please provide dietary preferences or health goals.")

        logger.debug(f"Received diet plan inputs - Preferences: '{preferences[:50]}...', Goals: '{goals[:50]}...'")

        # --- Construct Prompt for Gemini ---
        # Match the prompt style from the frontend JS (plain English, short plan)
        prompt = f"""
        Suggest a brief, simple diet plan based on the following user information.
        Provide the plan in plain English text only. Do not use bullet points, bolding, italics, or any special formatting symbols.
        Keep the plan concise and easy to follow.

        User Information:
        - Dietary Preferences/Restrictions: {preferences if preferences else 'None specified'}
        - Health Goals: {goals if goals else 'None specified'}

        Suggested Diet Plan:
        """

        # --- Call Gemini API ---
        logger.info("Calling Gemini API for diet plan suggestion...")
        diet_plan_text = await call_gemini_api(prompt)

        # --- Process Result ---
        if diet_plan_text:
            # Basic cleanup: remove potential leading phrases if needed
            cleaned_plan = diet_plan_text.split(':')[-1].strip()
            logger.info(f"Gemini suggested diet plan ({len(cleaned_plan)} chars): '{cleaned_plan[:150]}...'")
            return jsonify({"diet_plan": cleaned_plan}), 200
        else:
            logger.error("Gemini API call for diet plan failed or returned no content.")
            raise InternalServerError("Could not generate diet plan suggestion at this time.")

    except BadRequest as e:
        logger.warning(f"Diet Plan - Bad Request ({e.code}): {e.description}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during diet plan generation: {e}", exc_info=True)
        raise InternalServerError("An unexpected error occurred while generating the diet plan.")

# Remember to register this blueprint in your main Flask app factory or app instance
# Example (in your __init__.py or app.py):
# from .routes.suggestions_route import suggestions_bp
# app.register_blueprint(suggestions_bp, url_prefix='/api')