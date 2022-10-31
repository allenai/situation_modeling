import logging
import gzip
from functools import wraps
import time
import pickle

def log_time(func):
    """This decorator prints the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        task = func.__name__
        args[0].logger.info("Finished executing %s in %s seconds" %\
                                (task,str(end-start)))
        return result
    return wrapper

class BaseClass(object):
    NAME = None

class LoggableClass(object):
    """Base class that allows for logging"""

    @property
    def logger(self):
        """Returns a logger instance

        """
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

class ConfigurableClass(LoggableClass):

    @classmethod
    def from_config(cls,config):
        """Builds an instance from configuration 

        :param config: the global configuration 
        """
        raise NotImplementedError('Configuration loader not implemented for this class!')

    def save(self,output_path):
        raise NotImplementedError('Saver not implemented for this class!')

    @classmethod
    def load(cls,input_path):
        raise NotImplementedError('Loader not implemented for this class!')

    @log_time
    def dump(self,path):
        """Make a pickled backup of the class instance (use carefully and sparingly
        for large objects)

        :param path: the path to put the pickled file 
        :rtype: None 
        """
        self.logger.info('Pickling the instance...')
        out_path = path if ".gz" in path else path+".gz"

        with gzip.open(out_path,'wb') as my_pickle:
            pickle.dump(self,my_pickle)

class LoadableClass(ConfigurableClass):
    def __enter__(self):
        raise NotImplementedError 
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

