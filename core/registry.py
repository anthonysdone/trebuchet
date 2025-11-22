OPS = {}
LAYERS = {}
OPTIMS = {}
SCHEDULERS = {}

def register_op(name):
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco

def register_layer(name):
    def deco(cls):
        LAYERS[name] = cls
        return cls
    return deco

def register_optim(name):
    def deco(cls):
        OPTIMS[name] = cls
        return cls
    return deco

def register_scheduler(name):
    def deco(cls):
        SCHEDULERS[name] = cls
        return cls
    return deco