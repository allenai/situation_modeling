from .utils import Register
import imp

def load_external_project(module_path):
    """Load an external set of custom python modules from file 
    and register them into `situation_modeling` pipeline. 

    :param module_path: the path of the target module 
    """
    f, filename, description = imp.find_module(module_path)
    example_package = imp.load_module(module_path, f, filename, description)
