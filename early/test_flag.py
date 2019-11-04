import numpy as np
from time import time
import sys

from opencl_handler import OpenCLHandler
import pyopencl as cl
import pyopencl.array as cl_array

from environment import Environment, prison, street

mf = cl.mem_flags

class Camera(object):
    def __init__(self):
        env_dim = 50
        self._width = 160
        self._height = 100
        field_of_view = 0.5 * np.pi
        propagation_length = 0.5
        position = [2, 2, env_dim//2]
        view_direction = [0, 0]
        walk_direction = 0

        environment = Environment()
        environment.load(prison, env_dim)

        opencl = OpenCLHandler('voodooray_2d.cl', 2)

        # opencl stuff
        # opencl buffers
        self._width_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._width))
        self._height_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._height))
        self._env_dim_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_dim))
        self._environment_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                        hostbuf=np.uint32(environment._env_array))
        #self._seed_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(seed))

        # opencl arrays
        self._pos_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._dir_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._distance_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height), dtype=np.float32)
        self._intensity_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height), dtype=np.uint8)

        pos_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(position))
        dir_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(view_direction))
        field_of_view_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(field_of_view))
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(propagation_length))


        opencl.prg.lidar(
            opencl.queue,
            (self._width, self._height),
            None,
            self._width_g,
            self._height_g,
            pos_in_g,
            dir_in_g,
            self._pos_out_a.data,
            self._dir_out_a.data,
            field_of_view_g,
            propagation_length_g,
            self._env_dim_g,
            self._environment_g,
            self._distance_a.data
        )

if __name__ == '__main__':
    camera = Camera()
    pass