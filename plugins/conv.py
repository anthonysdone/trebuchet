from core import device
from core.registry import register_op, register_layer
from core.tensor import Tensor
from core.layer import Layer

def im2col(x, kernel_h, kernel_w, stride, padding):
    N, C, H, W = x.shape

    if padding > 0: 
        x = device.xp.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
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

    device.xp.add.at(x_padded, (slice(None), k, i, j), cols) # type: ignore

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