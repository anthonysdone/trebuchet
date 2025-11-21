from . import device
from .registry import register_optim

class Optimizer:
    def __init__(self, params):
        self.params = params
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

@register_optim("sgd")
class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [device.xp.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: 
                continue

            grad = p.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data
            
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                update = self.velocity[i]
            else:
                update = grad
            
            p.data = p.data - self.lr * update

        self.zero_grad()

@register_optim("adam")
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [device.xp.zeros_like(p.data) for p in self.params]
        self.v = [device.xp.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            p.data = p.data - self.lr * m_hat / (device.xp.sqrt(v_hat) + self.eps)

        self.zero_grad()
