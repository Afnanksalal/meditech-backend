# app/services/audio_processing.py
import subprocess
import logging
from pathlib import Path
from flask import current_app

logger = logging.getLogger(__name__)

def convert_audio(input_path: Path, output_path: Path) -> bool:
    """
    Converts audio file to WAV format (mono, 16kHz) using FFmpeg.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the converted WAV file.

    Returns:
        True if conversion was successful, False otherwise.
    """
    ffmpeg_path = current_app.config['FFMPEG_PATH']
    target_sr = current_app.config['TARGET_SAMPLE_RATE']

    try:
        input_path_str = str(input_path)
        output_path_str = str(output_path)

        cmd = [
            ffmpeg_path, '-y', '-i', input_path_str,
            '-ar', str(target_sr), '-ac', '1', '-f', 'wav',
            output_path_str
        ]
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")

        process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')

        if process.returncode != 0:
            # Log more detailed error info
            logger.error(f"FFmpeg failed (code {process.returncode}) for {input_path_str}.")
            logger.error(f"FFmpeg stdout:\n{process.stdout}")
            logger.error(f"FFmpeg stderr:\n{process.stderr}")
            return False
        else:
            if not output_path.exists() or output_path.stat().st_size == 0:
                 logger.error(f"FFmpeg reported success, but output file is missing or empty: {output_path_str}")
                 logger.error(f"FFmpeg stdout:\n{process.stdout}")
                 logger.error(f"FFmpeg stderr:\n{process.stderr}")
                 return False
            logger.info(f"Audio conversion successful: {input_path_str} -> {output_path_str}")
            return True
    except FileNotFoundError:
        logger.error(f"FFmpeg executable not found at '{ffmpeg_path}'. Please ensure it's installed and in the system PATH or update FFMPEG_PATH in config.")
        return False
    except Exception as e:
        logger.error(f"Error during FFmpeg processing for {input_path}: {e}", exc_info=True)
        return False