import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from imageio import imsave
import pyopencl.array as cl_array
from pygame import surfarray

mf = cl.mem_flags

class Camera(object):
    def __init__(self, position, view_direction, walk_direction, width, height, opencl, environment):
        self._position = position
        self._view_direction = view_direction
        self._walk_direction = walk_direction
        self._width = width
        self._height = height
        self._trace = False

        seed = np.random.randint(-(2 ** 31), 2 ** 31 - 1, size=(self._width, self._height))

        # opencl stuff
        # opencl buffers
        self._width_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._width))
        self._height_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._height))
        self._env_dim_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_dim))
        self._environment_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array))
        self._seed_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(seed))

        # opencl arrays
        self._pos_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._dir_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._distance_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height), dtype=np.float32)
        self._intensity_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.uint8)

    def update_environment_buffer(self, opencl, environment):
        self._environment_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint8(environment._env_array))

    def move(self, magnitude, environment):
        new_position = [
            self._position[0] + np.cos(self._walk_direction) * magnitude,
            self._position[1] + np.sin(self._walk_direction) * magnitude,
            self._position[2]
        ]

        if new_position >= [0, 0, 0] and new_position < environment._env_dim:
            if environment._env_array[int(new_position[0])][int(new_position[1])][int(new_position[2])] == 0:
                self._position = new_position

    def set_view_direction(self, x, y):
        rel_phi = x / self._width
        rel_theta = y / self._height
        self._view_direction[0] = (rel_phi - 0.5) * np.pi
        self._view_direction[1] = (-rel_theta + 0.5) * 0.8 * np.pi

    def rotate_walk_direction(self, delta):
        self._walk_direction += delta

    def get_position(self):
        return self._position

    def get_direction(self):
        return [self._view_direction[0] + self._walk_direction, self._view_direction[1]]

    def switch(self):
        self._trace = not (self._trace)

    def lidar(self, opencl, field_of_view, propagation_length):
        pos_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_position()))
        dir_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_direction()))
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
            self._distance_a.data,
        )

    def trace(self, opencl, field_of_view, propagation_length):
        pos_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_position()))
        dir_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_direction()))
        field_of_view_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(field_of_view))
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(propagation_length))

        opencl.prg.trace(
            opencl.queue,
            (self._width, self._height),
            None,
            propagation_length_g,
            self._width_g,
            self._height_g,
            pos_in_g,
            dir_in_g,
            field_of_view_g,
            self._intensity_a.data,
            self._env_dim_g,
            self._environment_g,
            self._seed_g
        )

    def fill_surface(self, surface):
        if self._trace:
            values = self.get_intensity()
        else:
            d = 255.0 / (self.get_distance()**2 + 1.0)
            values = np.dstack([d, d, d])
        surfarray.blit_array(surface, values)

    def snapshot(self, opencl,field_of_view, propagation_length, nr_of_samples=1000):
        # TODO: refactor with new output structure
        values_out = np.zeros((self._width, self._height, 3), dtype=np.uint32)
        for sample in range(nr_of_samples):
            self.trace(opencl, field_of_view, propagation_length)
            values_out += self.get_intensity()

        imsave('snapshot.png', values_out.transpose(1, 0, 2))


    def get_intensity(self):
        return self._intensity_a.get()

    def get_distance(self):
        return self._distance_a.get()
