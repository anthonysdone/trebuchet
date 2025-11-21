import numpy as np

cp = None
jnp = None

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

try:
    import jax.numpy as jnp  # type: ignore
except Exception:
    jnp = None

xp = np

def use_cpu():
    global xp
    xp = np

def use_gpu(): 
    global xp
    if cp is not None:
        xp = cp
    elif jnp is not None:
        xp = jnp
    else:
        raise ImportError("[Device]: CuPy and JAX are not installed.")