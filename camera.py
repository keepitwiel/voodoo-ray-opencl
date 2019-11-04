import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from imageio import imsave
import pyopencl.array as cl_array
from pygame import surfarray
import hashlib

mf = cl.mem_flags

FISH_EYE = 0
FLAT_VIEW = 1
INFINITE = 2

TRACE = 0
TRACE_OLD = 1
LIDAR = 2

class Camera(object):
    def __init__(self, position, view_direction, walk_direction, width, height, opencl, environment):
        self._position = position
        self._view_direction = view_direction
        self._walk_direction = walk_direction
        self._width = width
        self._height = height
        self._env_dim = environment._env_array.shape
        self._camera_style = FISH_EYE
        self._field_of_view = 1/3 * np.pi
        self._ray_spacing = 0.1
        self._mode = TRACE

        seed = np.random.randint(-(2 ** 31), 2 ** 31 - 1, size=(self._width, self._height))
        seed3d = np.random.randint(-(2 ** 31), 2 ** 31 - 1, size=(self._env_dim[0], self._env_dim[1], self._env_dim[2]))

        # opencl stuff
        # opencl buffers
        self._width_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._width))
        self._height_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._height))
        self._env_dim_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array.shape))
        self._env_dim_x_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array.shape[0]))
        self._env_dim_y_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array.shape[1]))
        self._env_dim_z_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array.shape[2]))
        self._nr_of_sides_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(6))
        self._environment_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint64(environment._env_array))
        self._seed_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(seed))
        self._seed3d_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(seed3d))

        # opencl arrays
        self._pos_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._dir_out_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.float32)
        self._distance_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height), dtype=np.float32)
        self._intensity_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 3), dtype=np.uint8)
        self._surface_id_a = cl_array.zeros(opencl.queue, shape=(self._width, self._height, 4), dtype=np.uint8)
        self._surface_rendered_a = 127 + cl_array.zeros(
            opencl.queue, shape=(self._env_dim[0], self._env_dim[1], self._env_dim[2], 6, 3), dtype=np.uint8
        )

    def update_environment_buffer(self, opencl, environment):
        self._environment_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint64(environment._env_array))
        self._env_dim_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(environment._env_array.shape))

    def move(self, magnitude, environment, opencl):
        new_position = (
            self._position[0] + np.cos(self._view_direction[1]) * np.cos(self._view_direction[0]) * magnitude,
            self._position[1] + np.cos(self._view_direction[1]) * np.sin(self._view_direction[0]) * magnitude,
            self._position[2] + np.sin(self._view_direction[1]) * magnitude
        )

        if new_position >= (0, 0, 0) and new_position < environment._env_array.shape:
            if environment._env_array[int(new_position[0])][int(new_position[1])][int(new_position[2])] == 0:
                self._position = new_position

        print('new position at {}, {}, {}'.format(self._position[0], self._position[1], self._position[2]))

    def set_view_direction(self, x, y, opencl, propagation_length):
        rel_phi = x / self._width
        rel_theta = y / self._height
        self._view_direction[0] = (rel_phi - 0.5) * 2.0 * np.pi
        self._view_direction[1] = (-rel_theta + 0.5) * 0.8 * np.pi


    def rotate_walk_direction(self, delta, opencl):
        self._walk_direction += delta


    def get_position(self):
        return self._position

    def get_direction(self):
        return [self._view_direction[0] + self._walk_direction, self._view_direction[1]]

    def switch(self):
        if self._mode == TRACE:
            self._mode = TRACE_OLD
        elif self._mode == TRACE_OLD:
            self._mode = LIDAR
        else:
            self._mode = TRACE
        print(f'switching render mode to {self._mode}...')


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

    def trace_old(self, opencl, field_of_view, propagation_length):
        pos_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_position()))
        dir_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_direction()))
        field_of_view_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(field_of_view))
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(propagation_length))

        opencl.prg.trace_old(
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

    def trace(self, opencl, propagation_length):
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(propagation_length))

        opencl.prg.trace(
            opencl.queue,
            (self._width, self._height),
            None,
            propagation_length_g,
            self._width_g,
            self._height_g,
            self._pos_out_a.data,
            self._dir_out_a.data,
            self._intensity_a.data,
            self._env_dim_g,
            self._environment_g,
            self._seed_g
        )

    def fill_surface(self, surface):
        if self._mode in [TRACE, TRACE_OLD]:
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
