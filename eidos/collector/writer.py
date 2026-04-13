import sqlite3
import threading
import queue
import time
import json
import os
import logging

class DatabaseWriter(threading.Thread):
    """
    A thread-safe SQLite writer that uses a queue to batch writes.
    Enables WAL mode for high-performance concurrent reads/writes.
    """
    def __init__(self, db_path: str, schema_path: str):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.schema_path = schema_path
        self.queue = queue.Queue()
        self.running = True
        
        # Ensure the DB starts correctly
        self._init_db()

    def _init_db(self):
        """Initializes the database with your schema.sql file if it doesn't exist."""
        db_exists = os.path.exists(self.db_path)
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        
        if not db_exists:
            logging.info(f"Initializing database at {self.db_path}")
            with open(self.schema_path, 'r') as f:
                conn.executescript(f.read())
        
        conn.commit()
        conn.close()

    def run(self):
        """The background loop that picks items from the queue and writes to disk."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        
        while self.running:
            try:
                # Wait for data for 1 second
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                
                msg_type, data = item
                
                if msg_type == "observation":
                    self._insert_observation(conn, data)
                elif msg_type == "session":
                    self._insert_session(conn, data)
                
                # Commit AND signal task completion for EVERY successful write
                conn.commit()
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"DatabaseWriter Error: {e}")

        conn.close()

    def add_observation(self, data: dict):
        """Used by other collectors to 'fire and forget' data."""
        self.queue.put(("observation", data))

    def create_session(self, session_uid: str, metadata: dict = None):
        """Used once at the start of every recording session."""
        self.queue.put(("session", {
            "session_uid": session_uid,
            "start_time": time.time(),
            "metadata": json.dumps(metadata) if metadata else None
        }))

    def _insert_session(self, conn, data):
        query = "INSERT INTO sessions (session_uid, start_time, metadata) VALUES (:session_uid, :start_time, :metadata)"
        conn.execute(query, data)

    def _insert_observation(self, conn, data):
        query = """
        INSERT INTO observations (
            session_id, timestamp, frame_path, app_name, window_title, 
            keystrokes, mouse_x, mouse_y, mouse_events, audio_transcript
        ) VALUES (
            :session_id, :timestamp, :frame_path, :app_name, :window_title, 
            :keystrokes, :mouse_x, :mouse_y, :mouse_events, :audio_transcript
        )
        """
        # Defensive check: make sure lists aren't passed directly to SQLite
        if isinstance(data.get('keystrokes'), (list, dict)):
            data['keystrokes'] = json.dumps(data['keystrokes'])
        if isinstance(data.get('mouse_events'), (list, dict)):
            data['mouse_events'] = json.dumps(data['mouse_events'])
            
        conn.execute(query, data)

    def stop(self):
        """Signal the thread to finish up and close."""
        self.running = False
        self.queue.put(None)
