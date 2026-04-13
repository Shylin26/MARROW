import time
import uuid
import threading
import logging
from pathlib import Path
from eidos.collector.screen import ScreenCollector
from eidos.collector.audio import AudioCollector
from eidos.collector.input import InputCollector
from eidos.collector.accessibility import AccessibilityCollector
from eidos.collector.writer import DatabaseWriter


class MarrowEngine:
    """MarrowEngine is the control orchestrator. It starts all collectors and runs a loop to save data."""
    def __init__(self, data_dir="./data", db_path="./db/marrow.db", schema_path="./db/schema.sql"):
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        self.db_path = db_path
        self.schema_path = schema_path
        
        # Create the 'data/frames' folder if it is not present
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = DatabaseWriter(db_path, schema_path)
        self.screen_collector = ScreenCollector(str(self.frames_dir))
        self.audio_collector = AudioCollector()
        self.input_collector = InputCollector()
        self.accessibility_collector = AccessibilityCollector()
        self.running = False
        self.session_uid = str(uuid.uuid4())
    
    def start(self):
        logging.info(f"Starting the Marrow Session: {self.session_uid}")
        self.running = True
        self.writer.start()
        self.audio_collector.start()
        self.input_collector.start()
        self.writer.create_session(self.session_uid, {"env": "macOS"})
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def stop(self):
        logging.info("Stopping Marrow...")
        self.running = False
        self.audio_collector.stop()
        self.input_collector.stop()
        self.writer.stop()
        self.heartbeat_thread.join(timeout=2)
        logging.info("Marrow stopped successfully.")
    
    def _heartbeat_loop(self):
        """
        The core loop that runs every 2 seconds.
        It collects data from all 'passive' collectors and sends them to the writer.
        """
        logging.info("Heartbeat loop started")
        while self.running:
            try:
                start_time = time.time()
                frame_name = f"{self.session_uid}_{int(start_time)}"
                frame_path = self.screen_collector.capture(frame_name)
                app_info = self.accessibility_collector.get_info()
                input_data = self.input_collector.get_and_flush()
                transcript = self.audio_collector.get_latest_transcription()
                
                observation = {
                    "session_id": self.session_uid,
                    "timestamp": start_time,
                    "frame_path": frame_path,
                    "app_name": app_info.get("app_name"),
                    "window_title": app_info.get("window_title"),
                    "keystrokes": input_data.get("keystrokes"),
                    "mouse_x": input_data.get("mouse_x"),
                    "mouse_y": input_data.get("mouse_y"),
                    "mouse_events": input_data.get("mouse_events"),
                    "audio_transcript": transcript
                }
                
                self.writer.add_observation(observation)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, 2.0 - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logging.error(f"Error in engine heartbeat: {e}")
                time.sleep(1.0)




