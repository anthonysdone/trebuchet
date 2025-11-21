from . import device
from .tensor import Tensor
from .registry import register_layer
from . import ops

class Layer:
    def __init__(self):
        self.training = True

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError
    
    def parameters(self):
        return []

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

@register_layer("linear")
class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        limit = device.xp.sqrt(6 / (in_features + out_features))
        
        self.weight = Tensor(
            device.xp.random.uniform(-limit, limit, (in_features, out_features)), 
            req_grad=True)
        self.bias = (
            Tensor(device.xp.zeros(out_features), req_grad=True) if bias else None)
        
    def forward(self, x): 
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self):
        params = [self.weight]
        if self.bias is not None: 
            params.append(self.bias)
        return params
        
@register_layer("relu")
class Relu(Layer): 
    def __init__(self): 
        super().__init__()

    def forward(self, x):
        return ops.relu(x)
        
@register_layer("sigmoid")
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ops.sigmoid(x)

@register_layer("tanh")
class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ops.tanh(x)

@register_layer("softmax")
class Softmax(Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return ops.softmax(x, axis=self.axis)