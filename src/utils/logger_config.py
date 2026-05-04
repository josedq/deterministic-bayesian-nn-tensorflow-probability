import logging
import inspect
import os

def setup_logger(name=None):
    if name is None or name == "__main__":
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        caller_file = frame.filename
        if module and module.__package__:
            base = os.path.splitext(os.path.basename(caller_file))[0]
            name = f"{module.__package__}.{base}"
        else:
            name = os.path.splitext(os.path.basename(caller_file))[0]

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    return logger