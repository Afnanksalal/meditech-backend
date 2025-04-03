# app/sockets.py
from flask import request, current_app # Import current_app
from flask_socketio import emit, join_room, leave_room
import logging
from datetime import datetime

# Assuming socketio and active_rooms are defined and imported from your app's __init__ or main file
from . import socketio, active_rooms # Make sure these are correctly imported

logger = logging.getLogger(__name__)

# No changes needed for handle_connect, handle_disconnect

@socketio.on('join_room')
def handle_join_room(data):
    """Handles a user joining a room and sends WebRTC config."""
    room_id = data.get('room_id')
    username = data.get('username', f'User_{request.sid[:4]}')

    if not room_id:
        logger.warning(f"Join attempt failed: No room_id provided by {request.sid}")
        emit('error', {'message': 'Room ID is required to join.'})
        return

    if room_id not in active_rooms:
        logger.info(f"Room {room_id} not found. Creating it.")
        active_rooms[room_id] = {'users': {}}

    if request.sid in active_rooms[room_id].get('users', {}):
        logger.warning(f"User {request.sid} ('{username}') is already in room {room_id}.")
        # Optionally resend config if they somehow disconnected/reconnected without leaving fully
        try:
            webrtc_config = current_app.config.get('WEBRTC_CONFIG')
            if webrtc_config:
                 # Send config only to the joining/rejoining user
                 emit('webrtc_config', {'config': webrtc_config}, to=request.sid)
            else:
                 logger.error("WEBRTC_CONFIG is not defined in Flask app config!")
                 emit('error', {'message': 'Server configuration error for WebRTC.'}, to=request.sid)
        except Exception as e:
            logger.error(f"Error sending WebRTC config to {request.sid}: {e}")
            emit('error', {'message': 'Error retrieving server configuration.'}, to=request.sid)
        # Inform the user they are already joined
        emit('already_joined', {'room_id': room_id, 'username': active_rooms[room_id]['users'][request.sid]})
        return

    join_room(room_id)
    active_rooms[room_id].setdefault('users', {})[request.sid] = username
    logger.info(f"User {request.sid} ('{username}') joined room {room_id}")

    # --- Send WebRTC Configuration ---
    try:
        webrtc_config = current_app.config.get('WEBRTC_CONFIG')
        if webrtc_config:
            # Send config only to the joining user
            emit('webrtc_config', {'config': webrtc_config}, to=request.sid)
            logger.debug(f"Sent WebRTC config to {request.sid} for room {room_id}")
        else:
            logger.error("WEBRTC_CONFIG is not defined in Flask app config!")
            emit('error', {'message': 'Server configuration error for WebRTC.'}, to=request.sid)
    except Exception as e:
        logger.error(f"Error sending WebRTC config to {request.sid}: {e}")
        emit('error', {'message': 'Error retrieving server configuration.'}, to=request.sid)
    # --- End Send WebRTC Configuration ---


    # Notify the joining user about the room state
    emit('joined_room', {
        'room_id': room_id,
        'username': username,
        'your_sid': request.sid, # Good to send the user their own SID
        'users': list(active_rooms[room_id]['users'].values()) # Send current users list
    })

    # Notify other users in the room
    emit('user_joined', {
        'user_id': request.sid,
        'username': username
    }, to=room_id, include_self=False) # Exclude sender

# No changes needed for handle_leave_room

@socketio.on('signal')
def handle_signal(data):
    """Relays WebRTC signaling messages (offer, answer, candidate)."""
    room_id = data.get('room_id')
    signal_data = data.get('signal') # Renamed from 'signal' to avoid conflict
    # target_sid = data.get('target_sid') # Keep if direct messaging is needed later

    if not room_id or not signal_data:
        logger.warning(f"Invalid signal from {request.sid}: Missing room_id or signal data.")
        return

    if room_id not in active_rooms or request.sid not in active_rooms[room_id].get('users', {}):
        logger.warning(f"Signal received for non-existent room '{room_id}' or user {request.sid} not in room.")
        return

    # Determine signal type for logging (optional but helpful)
    signal_type = "unknown"
    if isinstance(signal_data, dict):
        if 'type' in signal_data:
            signal_type = signal_data['type'] # offer or answer
        elif 'candidate' in signal_data:
            signal_type = 'candidate'

    logger.info(f"Relaying '{signal_type}' signal in room {room_id} from {request.sid}")

    # Prepare payload to send to others
    payload = {
        'user_id': request.sid, # Identify the sender to the recipient
        'signal': signal_data
    }

    # Broadcast the signal to everyone else in the room
    emit('signal', payload, to=room_id, include_self=False) # Crucial: exclude self

# No changes needed for handle_message, handle_ai_results
@socketio.on('message')
def handle_message(data):
    """Handles chat messages."""
    room_id = data.get('room_id')
    message_text = data.get('message')
    timestamp = data.get('timestamp', datetime.utcnow().isoformat() + "Z")

    if not room_id or not message_text:
        logger.warning(f"Invalid message from {request.sid}: Missing room_id or message text.")
        return

    if room_id not in active_rooms or request.sid not in active_rooms[room_id].get('users', {}):
        logger.warning(f"Message received for non-active room {room_id} or user {request.sid} not in room.")
        return

    username = active_rooms[room_id]['users'][request.sid]
    logger.info(f"Message in room {room_id} from '{username}' ({request.sid}): {message_text[:50]}...")

    emit('message', {
        'user_id': request.sid,
        'username': username,
        'message': message_text,
        'timestamp': timestamp
    }, to=room_id)

@socketio.on('ai_results')
def handle_ai_results(data):
    """Handles broadcasting AI analysis results to all room participants."""
    room_id = data.get('room_id')
    results = data.get('results')

    if not room_id or not results:
        logger.warning(f"Invalid AI results from {request.sid}: Missing room_id or results.")
        return

    if room_id not in active_rooms:
        logger.warning(f"AI results received for non-active room {room_id} from {request.sid}.")
        return

    username = active_rooms[room_id]['users'].get(request.sid, 'Unknown')
    logger.info(f"Broadcasting AI results in room {room_id} from '{username}' ({request.sid})")

    # Broadcast to all participants in the room including the sender
    emit('ai_results', {
        'room_id': room_id,
        'results': results,
        'processed_by': username,
        'timestamp': datetime.utcnow().isoformat() + "Z"
    }, to=room_id)