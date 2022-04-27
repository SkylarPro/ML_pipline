from typing import Dict
class Args:
    """
    Parser args
    """
    def __init__(self,cfg:Dict):
        for k,v in cfg.items():
            setattr(self,k,v)
