import numpy as np

cp = None
mlx = None

try:
    import cupy as cp # type: ignore
except Exception:
    cp = None

try:
    import mlx.core as mlx # type: ignore
except Exception:
    mlx = None

xp = np

def use_cpu():
    global xp
    xp = np

def use_gpu(): 
    global xp
    if cp is not None:
        print("Using CuPy")
        xp = cp
    elif mlx is not None:
        print("Using MLX")
        xp = mlx
    else:
        print("CuPy and MLX not found, falling back to NumPy")
        xp = np

# Wrapper functions for random operations to handle backend differences
class RandomWrapper:
    @staticmethod
    def randn(*shape):
        if xp.__name__ == 'mlx.core':
            return mlx.random.normal(shape) # type: ignore
        else:
            return xp.random.randn(*shape)
    
    @staticmethod
    def uniform(low, high, size):
        if xp.__name__ == 'mlx.core':
            return mlx.random.uniform(low, high, size) # type: ignore
        else:
            return xp.random.uniform(low, high, size)
    
    @staticmethod
    def binomial(n, p, size):
        if xp.__name__ == 'mlx.core':
            mask = np.random.binomial(n, p, size)
            return mlx.array(mask) # type: ignore
        else:
            return xp.random.binomial(n, p, size)

random = RandomWrapper()

def add_at(array, indices, values):
    if xp.__name__ == 'mlx.core':
        result = array.copy() if hasattr(array, 'copy') else mlx.array(array) # type: ignore
        result[indices] = result[indices] + values
        return result
    else:
        xp.add.at(array, indices, values) # type: ignore
        return array