from core import device
from core.registry import register_op, register_layer
from core.tensor import Tensor
from core.layer import Layer

def im2col(x, kernel_h, kernel_w, stride, padding):
    N, C, H, W = x.shape

    if padding > 0: 
        x = device.xp.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode="constant", constant_values=0)
    
    H_out = (H + 2 * padding - kernel_h) // stride + 1
    W_out = (W + 2 * padding - kernel_w) // stride + 1

    i0 = device.xp.repeat(device.xp.arange(kernel_h), kernel_w)
    i0 = device.xp.tile(i0, C)
    i1 = stride * device.xp.repeat(device.xp.arange(H_out), W_out)

    j0 = device.xp.tile(device.xp.arange(kernel_w), kernel_h * C)
    j1 = stride * device.xp.tile(device.xp.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = device.xp.repeat(device.xp.arange(C), kernel_h * kernel_w).reshape(-1, 1)

    cols = x[:, k, i, j]
    cols = device.xp.transpose(cols, (0, 2, 1))
    cols = cols.reshape(N * H_out * W_out, -1)

    return cols, (N, H_out, W_out)

def col2im(cols, x_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = x_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    H_out = (H + 2 * padding - kernel_h) // stride + 1
    W_out = (W + 2 * padding - kernel_w) // stride + 1

    cols = cols.reshape(N, H_out * W_out, -1)
    cols = device.xp.transpose(cols, (0, 2, 1))

    x_padded = device.xp.zeros((N, C, H_padded, W_padded))

    i0 = device.xp.repeat(device.xp.arange(kernel_h), kernel_w)
    i0 = device.xp.tile(i0, C)
    i1 = stride * device.xp.repeat(device.xp.arange(H_out), W_out)

    j0 = device.xp.tile(device.xp.arange(kernel_w), kernel_h * C)
    j1 = stride * device.xp.tile(device.xp.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = device.xp.repeat(device.xp.arange(C), kernel_h * kernel_w).reshape(-1, 1)

    x_padded = device.add_at(x_padded, (slice(None), k, i, j), cols)

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

@register_op("conv2d")
def conv2d(x, weight, bias=None, stride=1, padding=0):
    N, C_in, H, W = x.data.shape
    C_out, _, K_h, K_w = weight.data.shape

    x_col, (N, H_out, W_out) = im2col(x.data, K_h, K_w, stride, padding)
    w_row = weight.data.reshape(C_out, -1)

    out_data = x_col @ w_row.T

    if bias is not None: 
        out_data += bias.data
    
    out_data = out_data.reshape(N, H_out, W_out, C_out)
    out_data = device.xp.transpose(out_data, (0, 3, 1, 2))

    out = Tensor(out_data)
    out.req_grad = x.req_grad or weight.req_grad or (bias is not None and bias.req_grad)
    out.op = "conv2d"
    out.parents = {x, weight}
    if bias is not None:
        out.parents.add(bias)

    def backward():
        grad_out_reshaped = device.xp.transpose(out.grad, (0, 2, 3, 1)) # type: ignore
        grad_out_reshaped = grad_out_reshaped.reshape(N * H_out * W_out, C_out)

        if x.req_grad: 
            if x.grad is None: 
                x.grad = device.xp.zeros_like(x.data)
            dx_col = grad_out_reshaped @ w_row
            dx = col2im(dx_col, x.data.shape, K_h, K_w, stride, padding)
            x.grad += dx

        if weight.req_grad: 
            if weight.grad is None: 
                weight.grad = device.xp.zeros_like(weight.data)
            dw = grad_out_reshaped.T @ x_col
            weight.grad += dw.reshape(weight.data.shape)

        if bias is not None and bias.req_grad:
            if bias.grad is None:
                bias.grad = device.xp.zeros_like(bias.data)
            bias.grad += device.xp.sum(out.grad, axis=(0, 2, 3)) # type: ignore
        
    out._backward = backward
    return out

@register_op("max_pool2d")
def max_pool2d(x, kernel_size=2, stride=None): 
    if stride is None: 
        stride = kernel_size
    
    N, C, H, W = x.data.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1

    x_col, _ = im2col(x.data, kernel_size, kernel_size, stride, 0)
    x_col = x_col.reshape(N, H_out, W_out, C, kernel_size * kernel_size)
    x_col = device.xp.transpose(x_col, (0, 3, 1, 2, 4))

    out_data = device.xp.max(x_col, axis=4)
    max_indices = device.xp.argmax(x_col, axis=4)

    out = Tensor(out_data)
    out.req_grad = x.req_grad
    out.op = "max_pool2d"
    out.parents = {x}

    def backward():
        if x.req_grad: 
            if x.grad is None: 
                x.grad = device.xp.zeros_like(x.data)

            grad_col = device.xp.zeros_like(x_col)
            device.xp.put_along_axis(grad_col, max_indices[..., None], out.grad[..., None], axis=4) # type: ignore

            grad_col = device.xp.transpose(grad_col, (0, 2, 3, 1, 4))
            grad_col = grad_col.reshape(N * H_out * W_out, -1)

            dx = col2im(grad_col, x.data.shape, kernel_size, kernel_size, stride, 0)
            x.grad += dx
    
    out._backward = backward
    return out

@register_op("flatten")
def flatten(x, start_dim=1): 
    original_shape = x.data.shape
    new_shape = original_shape[:start_dim] + (-1,)
    out_data = x.data.reshape(new_shape)

    out = Tensor(out_data)
    out.req_grad = x.req_grad
    out.op = "flatten"
    out.parents = {x}

    def backward():
        if x.req_grad: 
            if x.grad is None: 
                x.grad = device.xp.zeros_like(x.data)
            x.grad += out.grad.reshape(original_shape) # type: ignore
    
    out._backward = backward
    return out

@register_op("batch_norm2d")
def batch_norm2d(x, gamma, beta, running_mean, running_var, eps, momentum, training):
    if training: 
        mean = device.xp.mean(x.data, axis=(0, 2, 3), keepdims=False)
        var = device.xp.var(x.data, axis=(0, 2, 3), keepdims=False)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean
        var = running_var

    mean_reshaped = mean.reshape(1, -1, 1, 1, 1)
    var_reshaped = var.reshape(1, -1, 1, 1)
    gamma_reshaped = gamma.data.reshape(1, -1, 1, 1)
    beta_reshaped = beta.data.reshape(1, -1, 1, 1)

    x_norm = (x.data - mean_reshaped) / device.xp.sqrt(var_reshaped + eps)
    out_data = gamma_reshaped * x_norm + beta_reshaped

    out = Tensor(out_data)
    out.req_grad = x.req_grad or gamma.req_grad or beta.req_grad
    out.op = "batch_norm2d"
    out.parents = {x, gamma, beta}

    def backward():
        N, C, H, W = x.data.shape
        NHW = N * H * W

        if beta.req_grad: 
            if beta.grad is None: 
                beta.grad = device.xp.zeros_like(beta.data)
            beta.grad += device.xp.sum(out.grad, axis=(0, 2, 3)) # type: ignore
        
        if gamma.req_grad: 
            if gamma.grad is None:
                gamma.grad = device.xp.zeros_like(gamma.data)
            gamma.grad += device.xp.sum(out.grad * x_norm, axis=(0, 2, 3))
        
        if x.req_grad:
            if x.grad is None:
                x.grad = device.xp.zeros_like(x.data)

            dx_norm = out.grad * gamma_reshaped
            std_inv = 1.0 / device.xp.sqrt(var_reshaped + eps)
            dx_norm_sum = device.xp.sum(dx_norm, axis=(0, 2, 3), keepdims=True)
            dx_norm_x_norm_sum = device.xp.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True)

            x.grad += std_inv / NHW * (NHW * dx_norm - dx_norm_sum - x_norm * dx_norm_x_norm_sum)

    out._backward = backward
    return out

@register_op("dropout")
def dropout(x, p, training): 
    if training and p > 0:
        mask = device.random.binomial(1, 1 - p, x.data.shape) / (1 - p)
        out = Tensor(x.data * mask)
        out.req_grad = x.req_grad
        out.op = "dropout"
        out.parents = {x}
    
        def backward():
            if x.req_grad:
                if x.grad is None:
                    x.grad = device.xp.zeros_like(x.data)
                x.grad += out.grad * mask
        
        out._backward = backward
        return out
    return x

@register_op("local_response_norm")
def local_response_norm(x, size, alpha, beta, k):
    N, C, H, W = x.data.shape
    pad_size = size // 2

    x_squared = x.data ** 2
    x_padded = device.xp.pad(x_squared, ((0,0), (pad_size, pad_size), (0,0), (0,0)), mode="constant")

    sum_squares = device.xp.zeros_like(x.data)
    for i in range(size):
        sum_squares += x_padded[:, i:i+C, :, :]
    
    scale = (k + alpha * sum_squares) ** beta
    out_data = x.data / scale

    out = Tensor(out_data)
    out.req_grad = x.req_grad
    out.op = "local_response_norm"
    out.parents = {x}

    def backward():
        if x.req_grad:
            if x.grad is None: 
                x.grad = device.xp.zeros_like(x.data)

            scale_grad = -alpha * beta * (k + alpha * sum_squares)
            x.grad += out.grad / scale + 2 * out.grad * x.data * scale_grad / scale # type: ignore
        
    out._backward = backward
    return out

@register_op("adaptive_avg_pool2d")
def adaptive_avg_pool2d(x, output_size):
    N, C, H, W = x.data.shape
    H_out, W_out = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    stride_h = H // H_out
    stride_w = W // W_out
    kernel_h = H - (H_out - 1) * stride_h
    kernel_w = W - (W_out - 1) * stride_w

    x_col, _ = im2col(x.data, kernel_h, kernel_w, stride_h if stride_h == stride_w else stride_h, 0)
    x_col = x_col.reshape(N, H_out, W_out, C, kernel_h * kernel_w)
    out_data = device.xp.mean(x_col, axis=4)
    out_data = device.xp.transpose(out_data, (0, 3, 1, 2))

    out = Tensor(out_data)
    out.req_grad = x.req_grad
    out.op = "adaptive_avg_pool2d"
    out.parents = {x}

    def backward():
        if x.req_grad: 
            if x.grad is None: 
                x.grad = device.xp.zeros_like(x.data)

            pool_size = kernel_h * kernel_w
            grad_out_transposed = device.xp.transpose(out.grad, (0, 2, 3, 1)) # type: ignore

            grad_col = device.xp.repeat(grad_out_transposed[..., None], pool_size, axis=4) / pool_size
            grad_col = grad_col.reshape(N * H_out * W_out, -1)

            dx = col2im(grad_col, x.data.shape, kernel_h, kernel_w, stride_h if stride_h == stride_w else stride_h, 0)
            x.grad += dx
        
    out._backward = backward
    return out

@register_layer("conv2d")
class Conv2d(Layer): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True): 
        super().__init__()
        self.stride = stride
        self.padding = padding

        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        limit = device.xp.sqrt(2.0 / (in_channels * k * k))

        self.weight = Tensor(
            device.random.randn(out_channels, in_channels, k, k) * limit, 
            req_grad=True)
        self.bias = Tensor(device.xp.zeros(out_channels), req_grad = True) if bias else None

    def forward(self, x): 
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def parameters(self):
        params = [self.weight]
        if self.bias is not None: 
            params.append(self.bias)
        return params

@register_layer("max_pool2d")
class MaxPool2d(Layer): 
    def __init__(self, kernel_size, stride=None): 
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x): 
        return max_pool2d(x, self.kernel_size, self.stride)

@register_layer("adaptive_avg_pool2d")
class AdaptiveAvgPool2d(Layer): 
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    
    def forward(self, x):
        return adaptive_avg_pool2d(x, self.output_size)

@register_layer("flatten")
class Flatten(Layer): 
    def __init__(self, start_dim=1): 
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x): 
        return flatten(x, self.start_dim)
    
@register_layer("batch_norm2d")
class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Tensor(device.xp.ones(num_features), req_grad=True)
        self.beta = Tensor(device.xp.zeros(num_features), req_grad=True)
        self.running_mean = device.xp.zeros(num_features)
        self.running_var = device.xp.ones(num_features)

    def forward(self, x):
        return batch_norm2d(x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps, self.momentum, self.training)
    
    def parameters(self):
        return [self.gamma, self.beta]
    
@register_layer("dropout")
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return dropout(x, self.p, self.training)

@register_layer("local_response_norm")
class LocalResponseNorm(Layer):
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        return local_response_norm(x, self.size, self.alpha, self.beta, self.k)