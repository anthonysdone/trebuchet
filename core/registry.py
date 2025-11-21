OPS = {}
LAYERS = {}
OPTIMS = {}

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