import python_cuda as pc
import numpy as np


N = 10_000_000
p = np.pi

t1 = np.random.randn(N).astype(np.float32)
V1 = pc.Vector(t1)

t2 = np.random.randn(N).astype(np.float32)
V2 = pc.Vector(t2)

V1.host2device()
V2.host2device()

V3 = V1 + V2
t3 = t1 + t2

V3.device2host()

assert np.allclose(t3, V3.get_array())

V3 = V1 - V2
t3 = t1 - t2

V3.device2host()

assert np.allclose(t3, V3.get_array())


norm_gpu = V1.norm(p)
norm_cpu = np.sum(np.power(np.abs(t1), p))**(1/p)

assert np.allclose(norm_cpu, norm_gpu)