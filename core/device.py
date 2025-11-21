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