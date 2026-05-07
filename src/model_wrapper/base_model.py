from typing import Any

class BaseModelWrapper:
    def __init__(self):
        self.model: Any = None
        pass
    
    def prepare_inputs(self, episodes):
        pass
    
    def eval(self):
        pass
    
    def run(self, *args, **kwds):
        pass
    
    def run_fixed(self, *args, **kwds):
        pass
    
    def run_unfixed(self, *args, **kwds):
        pass
