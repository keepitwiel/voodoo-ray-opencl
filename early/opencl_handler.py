import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

mf = cl.mem_flags


class OpenCLHandler(object):
    def __init__(self, filename, device=0):

        # Get platforms, both CPU and GPU
        plat = cl.get_platforms()
        CPU = plat[0].get_devices()
        self.ctx = cl.Context([CPU[device]])
        print(CPU)

        # Create queue for each kernel execution
        self.queue = cl.CommandQueue(self.ctx)

        # Kernel function
        with open(filename) as f:
            src = f.readlines()
            src = ''.join(src)

        # Kernel function instantiation
        self.prg = cl.Program(self.ctx, src).build()
