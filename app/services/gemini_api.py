# app/services/gemini_api.py
import logging
import asyncio
import aiohttp
import re
from flask import current_app

logger = logging.getLogger(__name__)

async def call_gemini_api(prompt: str, retries: int = 2, delay: int = 2) -> str | None:
    """
    Calls the Google Gemini API asynchronously using aiohttp with retry logic.
    """
    api_url = current_app.config.get('GEMINI_API_URL')
    headers = current_app.config.get('GEMINI_HEADERS')
    gemini_config = current_app.config.get('GEMINI_CONFIG')
    timeout_seconds = current_app.config.get('GEMINI_TIMEOUT_SECONDS', 60)

    if not api_url:
         logger.error("Gemini API URL is not configured. Cannot make API call.")
         return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gemini_config
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(retries + 1):
            try:
                logger.info(f"Calling Gemini API (Attempt {attempt + 1}/{retries + 1})")
                async with session.post(api_url, headers=headers, json=payload) as response:
                    # Log status regardless of success/failure first
                    logger.debug(f"Gemini API response status: {response.status}")

                    if response.status != 200:
                         error_text = await response.text()
                         logger.error(f"Gemini API HTTP error: Status {response.status}, Body: {error_text[:500]}...") # Log start of body
                         response.raise_for_status() # Raise ClientResponseError

                    result = await response.json()
                    # logger.debug(f"Gemini raw response JSON: {result}") # Be careful logging potentially large/sensitive data

                    # --- Safer response parsing ---
                    candidates = result.get('candidates')
                    if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
                        # Check for prompt feedback indicating blockage
                        prompt_feedback = result.get('promptFeedback')
                        if prompt_feedback:
                             block_reason = prompt_feedback.get('blockReason')
                             safety_ratings = prompt_feedback.get('safetyRatings')
                             logger.error(f"Gemini call potentially blocked. Reason: {block_reason}, SafetyRatings: {safety_ratings}")
                             # Depending on the reason, maybe don't retry certain block types
                             # if block_reason == 'SAFETY' and attempt >= retries: break # Example
                        else:
                             logger.error(f"No valid 'candidates' list found in Gemini response: {result}")
                        # Continue to retry unless decided otherwise above
                        if attempt < retries: await asyncio.sleep(delay)
                        continue

                    first_candidate = candidates[0]
                    content = first_candidate.get('content')
                    if not content or not isinstance(content, dict):
                         logger.error(f"No valid 'content' dictionary in first candidate: {first_candidate}")
                         if attempt < retries: await asyncio.sleep(delay)
                         continue

                    parts = content.get('parts')
                    if not parts or not isinstance(parts, list) or len(parts) == 0:
                         logger.error(f"No valid 'parts' list in content: {content}")
                         if attempt < retries: await asyncio.sleep(delay)
                         continue

                    first_part = parts[0]
                    text = first_part.get('text')

                    if text is None: # Check specifically for None, as "" might be valid
                         logger.error(f"No 'text' found in first part: {first_part}")
                         if attempt < retries: await asyncio.sleep(delay)
                         continue
                    # --- End Safer parsing ---

                    logger.info("Gemini call successful.")
                    return text.strip()

            except aiohttp.ClientResponseError as e:
                # Already logged status and body above if status != 200
                logger.error(f"Gemini API HTTP error caught (Attempt {attempt + 1}): Status {e.status}, Message: {e.message}")
                # Retry logic for specific codes
                if e.status == 429 and attempt < retries: # Rate limit
                    retry_after = int(e.headers.get("Retry-After", delay * 2))
                    logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    delay = retry_after # Use server suggested delay if available
                    continue
                elif 500 <= e.status < 600 and attempt < retries: # Server-side error
                     logger.warning(f"Server error ({e.status}). Retrying after {delay} seconds...")
                     await asyncio.sleep(delay)
                     delay *= 2 # Exponential backoff
                     continue
                else: # Non-retriable client error or final attempt failed
                    logger.error("Non-retriable HTTP error or retries exhausted.")
                    return None # Failed after retries or non-retriable error
            except asyncio.TimeoutError:
                 logger.warning(f"Gemini API call timed out (Attempt {attempt + 1}) after {timeout_seconds}s.")
                 if attempt < retries:
                      await asyncio.sleep(delay)
                      delay *= 2
                      continue
                 else:
                      return None # Failed after retries
            except aiohttp.ClientError as e: # Catch other aiohttp client errors (connection issues)
                 logger.error(f"Gemini API connection error (Attempt {attempt + 1}): {e}", exc_info=True)
                 if attempt < retries:
                      await asyncio.sleep(delay)
                      delay *= 2
                      continue
                 else:
                      return None # Failed after retries
            except Exception as e:
                 logger.error(f"An unexpected error occurred during Gemini API call (Attempt {attempt + 1}): {e}", exc_info=True)
                 # Depending on the error, might want to break or continue retrying
                 if attempt >= retries: # If it's the last attempt, return None
                      return None
                 await asyncio.sleep(delay) # Wait before potential retry for unexpected errors


    logger.error("Gemini API call failed after all retries.")
    return None


async def translate_with_gemini(text: str) -> str:
    """Translates Malayalam text to English using the Gemini API asynchronously."""
    if not text:
        logger.warning("translate_with_gemini called with empty text.")
        return "Translation unavailable: No input text"

    prompt = f"""
    Translate the following Malayalam medical text to English accurately and completely.
    Do not summarize or omit any details. Preserve medical terminology.
    Maintain the original context and meaning. If the input text is already substantially in English, return it as is with minimal changes.

    Input Text:
    ---
    {text}
    ---

    English Translation:
    """
    logger.info("Requesting translation from Gemini.")
    translated = await call_gemini_api(prompt)

    if translated:
        logger.info("Translation received from Gemini.")
        # Basic post-processing
        translated = re.sub(r'^(english translation:|---)\s*', '', translated, flags=re.IGNORECASE | re.MULTILINE).strip()
        return translated
    else:
        logger.error("Translation failed or Gemini returned no content.")
        return "Translation unavailable"

async def extract_emr(text: str) -> dict:
    """Extracts EMR data from transcribed text using Gemini asynchronously."""
    if not text:
        logger.warning("extract_emr called with empty text.")
        return {}

    prompt = f"""
    Analyze the following transcribed medical text. Extract the key Electronic Medical Record (EMR) data points.
    Return the extracted information strictly as KEY: VALUE pairs, one pair per line.
    Do not include any introductory sentences, explanations, or extraneous text like "Extracted EMR Data:".

    Extract these specific fields:
    - Presenting Complaint:
    - History of Presenting Illness:
    - Past Medical History:
    - Current Medications:
    - Allergies:

    If a specific field is not mentioned in the text, return the key with the value "Not mentioned".
    Do not infer or make up information. Only extract explicitly stated details.

    Source Text:
    ---
    {text}
    ---
    """
    logger.info("Requesting EMR extraction from Gemini.")
    result = await call_gemini_api(prompt)

    emr_data = {}
    required_keys = [
        "Presenting Complaint", "History of Presenting Illness",
        "Past Medical History", "Current Medications", "Allergies"
    ]

    if result:
        logger.info("EMR data received from Gemini.")
        for line in result.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Only add if key is one we're looking for, case-insensitive check might be good
                normalized_key = key.replace(':', '').strip() # Basic normalization
                # Find the matching required key if possible
                matched_key = next((req_key for req_key in required_keys if req_key.lower() == normalized_key.lower()), None)
                if matched_key and matched_key not in emr_data: # Add only once
                    emr_data[matched_key] = value
                elif key in required_keys and key not in emr_data: # Allow exact match too
                     emr_data[key] = value
                else:
                     logger.debug(f"Ignoring unexpected or duplicate EMR key in Gemini output: '{key}'")
            elif line:
                 logger.warning(f"Ignoring unexpected non-key-value line in EMR output: '{line}'")
    else:
        logger.warning("EMR extraction failed or Gemini returned no content.")

    # Ensure all required keys are present, adding "Not mentioned" if missing
    for key in required_keys:
        if key not in emr_data:
            emr_data[key] = "Not mentioned"

    return emr_data


async def generate_suggestions(emr_data: dict) -> dict:
    """Generates medical suggestions based on EMR data using Gemini asynchronously."""
    if not emr_data or all(v.lower() == "not mentioned" for v in emr_data.values()):
        logger.warning("No significant EMR data provided to generate_suggestions.")
        return {}

    emr_string = "\n".join(f"- {k}: {v}" for k, v in emr_data.items() if v.lower() != "not mentioned")
    if not emr_string:
         logger.warning("Filtered EMR data is empty, cannot generate suggestions.")
         return {}

    prompt = f"""
    Based on the following summarized Electronic Medical Record (EMR) data, provide concise potential medical suggestions.
    Return the suggestions strictly as KEY: VALUE pairs, one pair per line.
    Do not include any introductory sentences, explanations, or extraneous text like "Medical Suggestions:".

    Provide suggestions for these categories if relevant based on the input:
    - Differential Diagnosis:
    - Further Investigations:
    - Potential Treatment Options:
    - Specialist Referrals (if applicable):
    - Follow-up Recommendations:

    EMR Data Summary:
    ---
    {emr_string}
    ---
    """
    logger.info("Requesting suggestions from Gemini.")
    result = await call_gemini_api(prompt)

    suggestions = {}
    required_suggestion_keys = [
        "Differential Diagnosis", "Further Investigations", "Potential Treatment Options",
        "Specialist Referrals (if applicable)", "Follow-up Recommendations"
    ]

    if result:
        logger.info("Suggestions received from Gemini.")
        for line in result.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().replace(':','') # Clean key
                value = value.strip()
                # Find matching required key
                matched_key = next((req_key for req_key in required_suggestion_keys if req_key.lower() == key.lower()), None)
                if matched_key and matched_key not in suggestions:
                     suggestions[matched_key] = value
                elif key in required_suggestion_keys and key not in suggestions:
                     suggestions[key] = value
                else:
                     logger.debug(f"Ignoring unexpected or duplicate suggestion key: '{key}'")
            elif line:
                 logger.warning(f"Ignoring unexpected non-key-value line in suggestions output: '{line}'")
    else:
        logger.warning("Suggestion generation failed or Gemini returned no content.")

    # Ensure all required suggestion keys are present, maybe with a default
    for key in required_suggestion_keys:
        if key not in suggestions:
             suggestions[key] = "Not specified" # Or "Consult physician"

    return suggestions