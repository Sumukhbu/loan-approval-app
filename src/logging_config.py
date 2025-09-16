
import logging, os
def configure_logging(log_path='logs/project.log'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    return logging.getLogger()
