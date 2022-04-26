class Args:
    def __init__(self,cfg):
        for k,v in cfg.items():
            setattr(self,k,v)
