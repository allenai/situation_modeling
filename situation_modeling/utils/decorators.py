import functools

class Register:
    
    def __init__(self,name):
        functools.update_wrapper(self, name)
        self.name = name

    def factory_update_method(self,class_to_register):
        raise NotImplementedError

    def __call__(self,*args,**kwargs):
        class_to_add = args[0]
        self.factory_update_method(class_to_add)
        return class_to_add
