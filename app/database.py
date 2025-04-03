# app/database.py
import mysql.connector
from mysql.connector import pooling, Error
from flask import current_app
import logging
import random
import string

logger = logging.getLogger(__name__)
db_pool = None

def init_db_pool(app_config):
    """Initializes the database connection pool."""
    global db_pool
    if db_pool:
        logger.info("Database pool already initialized.")
        return

    try:
        logger.info(f"Initializing database pool '{app_config['DB_POOL_NAME']}' "
                    f"for {app_config['DB_USER']}@{app_config['DB_HOST']}:{app_config['DB_PORT']}")
        db_pool = pooling.MySQLConnectionPool(
            pool_name=app_config['DB_POOL_NAME'],
            pool_size=app_config['DB_POOL_SIZE'],
            host=app_config['DB_HOST'],
            port=app_config['DB_PORT'],
            user=app_config['DB_USER'],
            password=app_config['DB_PASSWORD'],
            database=app_config['DB_NAME'],
            pool_reset_session=True,
            auth_plugin='mysql_native_password' # Explicitly set if needed
        )
        logger.info("Database connection pool initialized successfully.")
        # Test connection
        conn = get_db_connection()
        if conn:
            conn.close()
            logger.info("Successfully obtained and closed a test connection from the pool.")
        else:
             logger.error("Failed to get a test connection from the pool.")
             raise RuntimeError("Database pool test connection failed.")

    except Error as err:
        logger.error(f"FATAL: Failed to initialize database pool: {err}", exc_info=True)
        raise RuntimeError(f"Database pool initialization failed: {err}") from err

def get_db_connection():
    """Gets a connection from the pool."""
    global db_pool
    if not db_pool:
        logger.error("Database pool is not initialized!")
        raise RuntimeError("Database pool is not available.")

    try:
        conn = db_pool.get_connection()
        return conn
    except Error as err:
        logger.error(f"Error getting connection from pool: {err}", exc_info=True)
        return None # Indicate failure

def close_db_connection(conn):
    """Returns a connection to the pool (no-op if conn is None)."""
    if conn:
        try:
            conn.close()
        except Error as err:
             logger.error(f"Error returning connection to pool: {err}", exc_info=True)


def create_tables_if_not_exist():
    """Creates required database tables."""
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot create tables: Failed to get DB connection.")
        return
    cursor = None
    try:
        cursor = conn.cursor()
        logger.info("Checking/Creating database tables...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS room (
                id INT AUTO_INCREMENT PRIMARY KEY,
                code VARCHAR(10) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("Table 'room' created or already exists.")
    except Error as err:
        logger.error(f"Error during table creation: {err}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        close_db_connection(conn)

# --- Room Helper Functions ---

def generate_room_code(length=6):
    """Generates a unique random room code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def db_create_room(room_code):
    """Inserts a new room code into the database."""
    conn = get_db_connection()
    if not conn: return False
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO room (code) VALUES (%s)", (room_code,))
        conn.commit()
        logger.info(f"Room '{room_code}' created in database.")
        return True
    except Error as err:
        logger.error(f"Error inserting room '{room_code}' into DB: {err}")
        conn.rollback()
        return False
    finally:
        if cursor: cursor.close()
        close_db_connection(conn)

def db_check_room_exists(room_code):
    """Checks if a room code exists in the database."""
    conn = get_db_connection()
    if not conn: return False
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM room WHERE code = %s LIMIT 1", (room_code,))
        exists = cursor.fetchone() is not None
        return exists
    except Error as err:
        logger.error(f"Error checking room '{room_code}' in DB: {err}")
        return False
    finally:
        if cursor: cursor.close()
        close_db_connection(conn)