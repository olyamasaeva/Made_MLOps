import logging
import functools

def log(func):
    logging.basicConfig(level = logging.DEBUG)
    logger = logging.getLogger()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info("*********************************************")
            logger.info(f"reached function {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"leaving function {func.__name__}")
            logger.info("*********************************************")
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e
    return wrapper
