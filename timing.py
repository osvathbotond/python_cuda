import python_cuda as pc
import numpy as np
import time


def single_add_math_only():
    N = 100_000_000

    T1 = np.random.randn(N).astype(np.float32)
    V1 = pc.Vector(T1, copy=True)

    T2 = np.random.randn(N).astype(np.float32)
    V2 = pc.Vector(T2, copy=True)

    
    V1.host2device()
    V2.host2device()
    t1_gpu = time.perf_counter()
    V3 = V1 + V2
    t2_gpu = time.perf_counter()
    V3.device2host()

    t1_cpu = time.perf_counter()
    T3 = T1 + T2
    t2_cpu = time.perf_counter()

    print(f"Single add on gpu (math only): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"Single add on cpu (math only): {1000*(t2_cpu-t1_cpu):.3f} ms")

def single_add_total():
    N = 100_000_000

    T1 = np.random.randn(N).astype(np.float32)
    V1 = pc.Vector(T1, copy=True)

    T2 = np.random.randn(N).astype(np.float32)
    V2 = pc.Vector(T2, copy=True)

    t1_gpu = time.perf_counter()
    V1.host2device()
    V2.host2device()
    V3 = V1 + V2
    V3.device2host()
    t2_gpu = time.perf_counter()

    t1_cpu = time.perf_counter()
    T3 = T1 + T2
    t2_cpu = time.perf_counter()

    print(f"Single add on gpu (total): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"Single add on cpu (total): {1000*(t2_cpu-t1_cpu):.3f} ms")

def norm_math_only():
    N = 100_000_000
    p = 3

    T = np.random.randn(N).astype(np.float32)
    V = pc.Vector(T, copy=True)

    
    V.host2device()
    t1_gpu = time.perf_counter()
    norm_gpu = V.norm(p)
    t2_gpu = time.perf_counter()
    V.device2host()

    t1_cpu = time.perf_counter()
    norm_cpu = np.linalg.norm(T, p)
    t2_cpu = time.perf_counter()

    print(f"{p}-norm on gpu (math only): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"{p}-norm on cpu (math only): {1000*(t2_cpu-t1_cpu):.3f} ms")

def norm_total():
    N = 100_000_000
    p = 3

    T = np.random.randn(N).astype(np.float32)
    V = pc.Vector(T, copy=True)

    t1_gpu = time.perf_counter()
    V.host2device()
    norm_gpu = V.norm(p)
    V.device2host()
    t2_gpu = time.perf_counter()

    t1_cpu = time.perf_counter()
    norm_cpu = np.linalg.norm(T, p)
    t2_cpu = time.perf_counter()

    print(f"{p}-norm on gpu (total): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"{p}-norm on cpu (total): {1000*(t2_cpu-t1_cpu):.3f} ms")

def longer_math_math_only():
    N = 100_000_000
    p = 4

    T1 = np.random.randn(N).astype(np.float32)
    V1 = pc.Vector(T1, copy=True)

    T2 = np.random.randn(N).astype(np.float32)
    V2 = pc.Vector(T2, copy=True)

    T3 = np.random.randn(N).astype(np.float32)
    V3 = pc.Vector(T3, copy=True)

    
    V1.host2device()
    V2.host2device()
    V3.host2device()
    t1_gpu = time.perf_counter()
    V4 = 2*V1 + V2
    V5 = 0.5*V1-1.1*V3
    r = V4.norm(p)
    V6 = r*V5
    t2_gpu = time.perf_counter()
    V6.device2host()

    t1_cpu = time.perf_counter()
    T4 = 2*T1 + T2
    T5 = 0.5*T1-1.1*T3
    r = np.linalg.norm(T4, p)
    T6 = r*T5
    t2_cpu = time.perf_counter()

    print(f"Longer math (math only): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"Longer math (math only): {1000*(t2_cpu-t1_cpu):.3f} ms")

def longer_math_total():
    N = 100_000_000
    p = 4

    T1 = np.random.randn(N).astype(np.float32)
    V1 = pc.Vector(T1, copy=True)

    T2 = np.random.randn(N).astype(np.float32)
    V2 = pc.Vector(T2, copy=True)

    T3 = np.random.randn(N).astype(np.float32)
    V3 = pc.Vector(T3, copy=True)

    
    t1_gpu = time.perf_counter()
    V1.host2device()
    V2.host2device()
    V3.host2device()
    V4 = 2*V1 + V2
    V5 = 0.5*V1-1.1*V3
    r = V4.norm(p)
    V6 = r*V5
    V6.device2host()
    t2_gpu = time.perf_counter()

    t1_cpu = time.perf_counter()
    T4 = 2*T1 + T2
    T5 = 0.5*T1-1.1*T3
    r = np.linalg.norm(T4, p)
    T6 = r*T5
    t2_cpu = time.perf_counter()

    print(f"Longer math (total): {1000*(t2_gpu-t1_gpu):.3f} ms")
    print(f"Longer math (total): {1000*(t2_cpu-t1_cpu):.3f} ms")

if __name__ == "__main__":
    single_add_math_only()
    single_add_total()
    norm_math_only()
    norm_total()
    longer_math_math_only()
    longer_math_total()