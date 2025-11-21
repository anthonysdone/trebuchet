from .device import xp
from .registry import register_op
from .tensor import Tensor

# =============================
# Elementwise
# =============================

def unbroadcast(grad, shape):
    ndim_diff = grad.ndim - len(shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)

    for i, (dim_orig, dim_grad) in enumerate(zip(shape, grad.shape)):
        if dim_orig == 1 and dim_grad > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad

@register_op("add")
def add(a, b):
    out = Tensor(a.data + b.data)
    out.req_grad = a.req_grad or b.req_grad
    out.op = "add"
    out.parents = {a, b}

    def backward():
        if a.req_grad:
            if a.grad is None:
                a.grad = xp.zeros_like(a.data)
            grad_a = unbroadcast(out.grad, a.data.shape)
            a.grad = a.grad + grad_a
            
        if b.req_grad:
            if b.grad is None:
                b.grad = xp.zeros_like(b.data)
            grad_b = unbroadcast(out.grad, b.data.shape)
            b.grad = b.grad + grad_b
    
    out._backward = backward
    return out

@register_op("sub")
def sub(a, b):
    out = Tensor(a.data - b.data)
    out.req_grad = a.req_grad or b.req_grad
    out.op = "sub"
    out.parents = {a, b}

    def backward():
        if a.req_grad:
            if a.grad is None:
                a.grad = xp.zeros_like(a.data)
            grad_a = unbroadcast(out.grad, a.data.shape)
            a.grad = a.grad + grad_a
            
        if b.req_grad:
            if b.grad is None:
                b.grad = xp.zeros_like(b.data)
            grad_b = unbroadcast(out.grad, b.data.shape)
            b.grad = b.grad - grad_b
    
    out._backward = backward
    return out

@register_op("mul")
def mul(a, b): 
    out = Tensor(a.data * b.data)
    out.req_grad = a.req_grad or b.req_grad
    out.op = "mul"
    out.parents = {a, b}

    def backward():
        if a.req_grad:
            if a.grad is None:
                a.grad = xp.zeros_like(a.data)
            grad_a = unbroadcast(out.grad * b.data, a.data.shape)
            a.grad = a.grad + grad_a
            
        if b.req_grad:
            if b.grad is None:
                b.grad = xp.zeros_like(b.data)
            grad_b = unbroadcast(out.grad * a.data, b.data.shape)
            b.grad = b.grad + grad_b
    
    out._backward = backward
    return out

# =============================
# Linear Algebra
# =============================

@register_op("matmul")
def matmul(a, b): 
    out = Tensor(a.data @ b.data)
    out.req_grad = a.req_grad or b.req_grad
    out.op = "matmul"
    out.parents = {a, b}

    def backward():
        if a.req_grad:
            if a.grad is None:
                a.grad = xp.zeros_like(a.data)
            a.grad = a.grad + out.grad @ b.data.T
        if b.req_grad:
            if b.grad is None:
                b.grad = xp.zeros_like(b.data)
            b.grad = b.grad + a.data.T @ out.grad
    
    out._backward = backward
    return out

# =============================
# Reductions
# =============================

@register_op("sum")
def sum(x, axis=None, keepdims=False): 
    out = Tensor(x.data.sum(axis=axis, keepdims=keepdims))
    out.req_grad = x.req_grad
    out.op = "sum"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = xp.zeros_like(x.data)
            x.grad = x.grad + out.grad * xp.ones_like(x.data)
    
    out._backward = backward
    return out

# =============================
# Activations
# =============================

@register_op("relu")
def relu(x):
    mask = x.data > 0
    out = Tensor(x.data * mask)
    out.req_grad = x.req_grad
    out.op = "relu"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = xp.zeros_like(x.data)
            x.grad = x.grad + out.grad * mask
    
    out._backward = backward
    return out

@register_op("sigmoid")
def sigmoid(x):
    out = Tensor(1 / (1 + xp.exp(-x.data)))
    out.req_grad = x.req_grad
    out.op = "sigmoid"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = xp.zeros_like(x.data)
            x.grad = x.grad + out.grad * (out.data * (1 - out.data))
    
    out._backward = backward
    return out

@register_op("tanh")
def tanh(x):
    out = Tensor(xp.tanh(x.data))
    out.req_grad = x.req_grad
    out.op = "tanh"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = xp.zeros_like(x.data)
            x.grad = x.grad + out.grad * (1 - out.data ** 2)
    
    out._backward = backward
    return out

@register_op("softmax")
def softmax(x, axis=-1):
    exps = xp.exp(x.data - xp.max(x.data, axis=axis, keepdims=True))
    smax = exps / xp.sum(exps, axis=axis, keepdims=True)

    out = Tensor(smax)
    out.req_grad = x.req_grad
    out.op = "softmax"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = xp.zeros_like(x.data)
            sum_term = xp.sum(out.grad * out.data, axis=axis, keepdims=True)
            x.grad = x.grad + out.data * (out.grad - sum_term)
            
    out._backward = backward
    return out

# =============================
# Loss Functions
# =============================

@register_op("mse_loss")
def mse_loss(pred, target):
    diff = pred.data - target.data

    out = Tensor(xp.mean(diff ** 2))
    out.req_grad = pred.req_grad
    out.op = "mse_loss"
    out.parents = {pred}

    def backward():
        if pred.req_grad:
            if pred.grad is None: 
                pred.grad = xp.zeros_like(pred.data)
            pred.grad = pred.grad + (2 * diff / diff.size) * out.grad

    out._backward = backward
    return out

@register_op("cross_entropy_loss")
def cross_entropy_loss(pred, target):
    logits_shifted = pred.data - xp.max(pred.data, axis=-1, keepdims=True)
    exp_logits = xp.exp(logits_shifted)
    softmax_probs = exp_logits / xp.sum(exp_logits, axis=-1, keepdims=True)

    if target.data.ndim == pred.data.ndim:
        loss_val = -xp.sum(target.data * xp.log(softmax_probs + 1e-8)) / pred.data.shape[0]
    else:
        batch_size = pred.data.shape[0]
        log_probs = xp.log(softmax_probs + 1e-8)
        loss_val = -xp.sum(log_probs[xp.arange(batch_size), target.data]) / batch_size

    out = Tensor(loss_val)
    out.req_grad = pred.req_grad
    out.op = "cross_entropy_loss"
    out.parents = {pred}

    def backward():
        if pred.req_grad:
            if pred.grad is None:
                pred.grad = xp.zeros_like(pred.data)

            batch_size = pred.data.shape[0]
            if target.data.ndim == pred.data.ndim:
                grad = (softmax_probs - target.data) / batch_size
            else:
                grad = softmax_probs.copy()
                grad[xp.arange(batch_size), target.data] -= 1
                grad = grad / batch_size
            pred.grad = pred.grad + grad * out.grad
    
    out._backward = backward
    return out

@register_op("bce_loss")
def bce_loss(pred, target):
    eps = 1e-8
    pred_clipped = xp.clip(pred.data, eps, 1-eps)

    loss_val = -xp.mean(
        target.data * xp.log(pred_clipped) 
        + (1 - target.data) * xp.log(1 - pred_clipped))

    out = Tensor(loss_val)
    out.req_grad = pred.req_grad
    out.op = "bce_loss"
    out.parents = {pred}

    def backward():
        if pred.req_grad:
            if pred.grad is None:
                pred.grad = xp.zeros_like(pred.data)

            grad = -(target.data / pred_clipped - (1 - target.data) / (1 - pred_clipped))
            grad = grad / pred.data.size
            pred.grad = pred.grad + grad * out.grad
    
    out._backward = backward
    return out