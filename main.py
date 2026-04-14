import logging
import time
import sys
from eidos.engine import MarrowEngine
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    engine=MarrowEngine(data_dir="./data",
    db_path="./db/marrow.db",
    schema_path="./db/schema.sql")
    try:
        engine.start()
        logging.info("Marrow is active. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Detected manual stop....")
    except Exception as e:
        logging.error(f"Critical error:{e}")
    finally:
        logging.info("Cleaning up and saving database...")
        engine.stop()
        logging.info("Shutdown complete.")

if __name__=="__main__":
    main()
    

    