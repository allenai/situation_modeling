import random
import json
import logging
from typing import List
from ..base import LoggableClass
from optparse import OptionParser,OptionGroup

util_logger = logging.getLogger('situation_modeling.readers.data_reader')


class DataReader(LoggableClass):
    """Base class for implementing dataset readers"""

    def __init__(self,data_list : List[str], evaluation=False):
        """Creates `DataReader` instance 

        :param data_list: the list of data instances 
        :param split: the target split
        """
        self._data_container = data_list
        self.evaluation = evaluation
        self.logger.info(
            'Loaded dataset instance, size=%s, evaluation=%s' %\
            (len(self),str(self.evaluation))
        )

    @property
    def data(self):
        return self._data_container
    
    @property
    def size(self):
        return len(self._data_container)

    def __iter__(self):
        yield self._data_container

    def __getitem__(self,i):
        return self._data_container[i]

    def __len__(self):
        return len(self._data_container)

    def __eq__(self,other):
        return self._data_container == other

    @staticmethod
    def _read(instance,evaluation=False):
        """Reads instances of Dataset

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls,path):
        """Load a reader from configuration 

        :param config: the global configuration 
        :returns: ReaderBase 
        """
        raise NotImplementedError
   
    @classmethod
    def from_instance(cls,reader_string):
        raise NotImplementedError

class JsonlReader(DataReader):
    """Reader for Json dataset"""

    @classmethod
    def json_file_reader(cls,path,evaluation,config):
        total_data = []
        
        with open(path) as my_json:
            for k,line in enumerate(my_json):
                line = line.strip()
                try:
                    json_line = json.loads(line)
                    line_instance = cls._read(json_line)
                    line_instance.evaluation = evaluation
                    total_data.append(line_instance)
                except json.JSONDecodeError as e:
                    util_logger.warn(f'Line {k} JSONDecodeError: {e}, skipping...')

                ## some logging to sanity check data
                if k < 3 and config and not config.quiet:
                    util_logger.info(
                        '\n ========\n raw line %d= \n \t%s \n processed line= \n \t %s \n ========' %\
                        (k,str(json_line),line_instance)
                    )

        return total_data 

    @classmethod
    def from_file(cls,path,config=None,evaluate=False):
        """Reads a dataset from file. If tokenizer if provided, 
        will try to tokenize it to gather together some information 
        about dataset (e.g., average length, etc.. to check truncation) 
        
        :param path: the target path
        :param config: the configuration (if provided)
        :param evaluate: run in evaluation mode
        """
        evaluation = False
        if '.jsonl' not in path: path += '.jsonl'

        ## for some classes this might involve adding additional output for
        ## doing computing metrics of some kind (e.g., exact match)
        #(config and (config.train_name+".jsonl" in path or config.dev_name+".jsonl" in path)) or evaluate:
        if ('dev.jsonl' in path or 'test.jsonl' in path) or (config and (config.dev_name+".jsonl" in path or config.test_name+".jsonl" in path)) or\
          evaluate is True:
            evaluation = True

        util_logger.info('Reading path=%s, evaluation=%s, config=%s' %\
                             (path,str(evaluation),True if config is not None else False)
        )

        ## read json
        total_data = cls.json_file_reader(path,evaluation,config)

        util_logger.info(
            'Read data, total number of instances=%d' % len(total_data)
        )

        ### max data (limit the training data)
        if config and len(total_data) > config.max_data and config.train_name+".jsonl" in path: 
            util_logger.info(
                f'Truncating data after shuffling, max_size={config.max_data}, seed={config.seed},path={path}'
            )
            random.shuffle(total_data)
            total_data = total_data[:config.max_data]
            util_logger.info('New size: %s' % str(len(total_data)))

        # max eval data
        elif config and len(total_data) > config.max_eval_data and config.dev_name+".jsonl" in path:
            util_logger.info(
                f'Truncating data after shuffling, max_size={config.max_eval_data}, seed={config.seed},path={path}'
            )
            random.shuffle(total_data)
            total_data = total_data[:config.max_eval_data]
            util_logger.info('New size: %s' % str(len(total_data)))
            
        return cls(total_data, evaluation)

    @classmethod
    def from_instance(cls,reader_string):
        """"Build a dataset from a single instance or string of instances

        :param reader_string: the input instance as a string 
        """
        json_rep = json.loads(reader_string)
        return cls([cls._read(json_rep)] )

def params(config):
    group = OptionGroup(config,"situation_modeling.readers.data_reader",
                            "General settings for data readers")

    config.add_option_group(group)
