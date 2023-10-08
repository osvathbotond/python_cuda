import python_cuda as pc
import numpy as np


N = 10_000_000
p = np.pi

t1 = np.random.randn(N).astype(np.float32)
V1 = pc.Vector(t1, copy=True)

assert V1.get_array() is not t1

t2 = np.random.randn(N).astype(np.float32)
V2 = pc.Vector(t2, copy=False)

assert V2.get_array() is t2

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

assert V1.get_array() is not t1
assert V2.get_array() is t2

norm_gpu = V1.norm(p)
norm_cpu = np.sum(np.power(np.abs(t1), p))**(1/p)

assert np.allclose(norm_cpu, norm_gpu)

V3 = 42*V1
t3 = 42*t1

V3.device2host()

assert np.allclose(t3, V3.get_array())

t4 = np.random.randn(N).astype(np.float32)
V4 = pc.Vector(t4, copy=False)

V4.host2device()

t4_orig = t4.copy()

V4.add(V1)

V4.device2host()

assert np.allclose(t4_orig+t1, V4.get_array())

V4.sub(V2)

V4.device2host()

assert np.allclose(t4_orig+t1-t2, V4.get_array())

V4.scale(2.1)

V4.device2host()

assert np.allclose((t4_orig+t1-t2)*2.1, V4.get_array())

assert V4.get_array() is t4