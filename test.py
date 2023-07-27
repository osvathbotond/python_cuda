import python_cuda as pc
import numpy as np

x1 = np.random.randn(100_000).astype(np.float32)
x2 = np.random.randn(100_000).astype(np.float32)
p = 4.2

assert np.allclose(x1+x2, pc.vector_add(x1, x2))
assert np.allclose(x1-x2, pc.vector_sub(x1, x2))
assert np.allclose(np.sum(np.power(np.abs(x1), p))**(1/p), pc.vector_norm(x1, p))