import threading
import sounddevice as sd 
import numpy as np
import whisper
import queue
import logging

class AudioCollector:
    def __init__(self, model_name="tiny"):
        self.model_name = model_name
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.running = False
        self.transcription = ""
        self._lock = threading.Lock()
    
    def start(self):
        """Initialise the AI model and start Background threads."""
        logging.info(f"Loading whisper model({self.model_name})...")
        self.model = whisper.load_model(self.model_name)
        self.running = True
        # Start the background thread that will transcript
        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _record_callback(self, indata, frames, time, status):
        """Every time the microphone has new data this runs"""
        if status:
            logging.warning(status)
        self.audio_queue.put(indata.copy())
    
    def _transcribe_loop(self):
        # Setting up the microphone stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._record_callback):
            while self.running:
                audio_data = []
                # Collect all available audio from the queue
                while not self.audio_queue.empty():
                    audio_data.append(self.audio_queue.get())
                
                if audio_data:
                    logging.debug("Transcribing audio chunk...")
                    concatenated = np.concatenate(audio_data).flatten()
                    result = self.model.transcribe(concatenated.astype(np.float32), fp16=False)
                    text = result.get("text", "").strip()
                    if text:
                        with self._lock:
                            self.transcription += " " + text
                
                # Sleep a bit to avoid hogging CPU while empty
                threading.Event().wait(1.0)
    
    def get_latest_transcription(self):
        with self._lock:
            latest = self.transcription.strip()
            self.transcription = ""
        return latest 
