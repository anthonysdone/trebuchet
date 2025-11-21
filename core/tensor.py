from . import device

class Tensor: 
    # =============================
    # Datatype
    # =============================

    def __init__(self, data, req_grad=False, op="", parents={}):
        self.data = device.xp.array(data)
        self.grad = None
        self.req_grad = req_grad

        self.op = op
        self.parents = parents
        self._backward = lambda: None
    
    # =============================
    # Helpers
    # =============================
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, req_grad={self.req_grad})"
    
    def zero_grad(self):
        self.grad = None
    
    def detach(self): 
        out = Tensor(self.data, req_grad = False)
        return out
    
    def numpy(self): 
        arr = self.data
        return arr
    
    # =============================
    # Operation Overloads
    # =============================
    
    def __add__(self, other): 
        from . import ops
        return ops.add(self, other)
    
    def __sub__(self, other):
        from . import ops
        return ops.sub(self, other)
    
    def __mul__(self, other):
        from . import ops
        return ops.mul(self, other)
    
    def __matmul__(self, other):
        from . import ops
        return ops.matmul(self, other)
    
    # =============================
    # Autograd
    # =============================
    
    def backward(self): 
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v.parents:
                    build(p)
                topo.append(v)
        
        build(self)

        self.grad = device.xp.ones_like(self.data)

        for v in reversed(topo):
            v._backward()