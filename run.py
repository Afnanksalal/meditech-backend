# run.py
import os
import logging
import sys

# --- NO MONKEY PATCHING ---
# Monkey patching section removed entirely

# --- Import Application Components ---
try:
    # Import Config to pass to factory, but async_mode is not used for patching
    from app.config import Config
    from app import create_app, socketio
except ImportError as e:
     print(f"ERROR: Failed to import from 'app' module: {e}", file=sys.stderr)
     print("Check imports within app/__init__.py and its submodules.", file=sys.stderr)
     sys.exit(1)
except Exception as e:
     print(f"ERROR: Unexpected error importing from 'app': {e}", file=sys.stderr)
     sys.exit(1)


# --- Create App and Run ---
app = create_app(Config) # Pass Config class/object
logger = app.logger # Get the logger configured by create_app

if __name__ == '__main__':
    host = app.config['HOST']
    port = app.config['PORT']
    debug = app.config['DEBUG']

    logger.info(f"Starting server on {host}:{port}...")
    logger.info(f"Debug mode: {debug}")
    # Log the effective async mode being used by SocketIO instance
    logger.info(f"SocketIO running with async_mode='{socketio.async_mode or 'Default (threading)'}'")
    logger.warning("Monkey patching is disabled. Using standard threading for SocketIO.")

    try:
        # Run using socketio.run() which handles the WSGI server correctly
        # For threading mode, the default Werkzeug development server is usually sufficient
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug, # Reloader is generally fine with threading mode
            # No need for allow_unsafe_werkzeug when monkey patching is off
        )
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")
    except Exception as e:
        # Log exceptions during server run
        logger.error(f"Server execution failed: {e}", exc_info=True)
        sys.exit(1) # Exit with error status