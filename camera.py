import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
from scipy.misc import imsave
import pyopencl.array as cl_array
from pygame import surfarray
import hashlib

mf = cl.mem_flags

FISH_EYE = 0
FLAT_VIEW = 1
INFINITE = 2


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
        self.trace_on = True

        seed = np.random.randint(-(2 ** 31), 2 ** 31 - 1, size=(self._width, self._height))
        random_phi = np.random.uniform(low=-np.pi, high=np.pi, size=(2**20, 1))
        random_theta = np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(2**20, 1))
        random_x = np.cos(random_phi) * np.cos(random_theta)
        random_y = np.sin(random_phi) * np.cos(random_theta)
        random_z = np.sin(random_theta)

        random_vector = np.concatenate([random_x, random_y, random_z], axis=1)

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
        self._random_vector_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(random_vector))
        self._random_index_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.uint32(0))

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
                self.calculate_ray_origins(opencl)

        print('new position at {}, {}, {}'.format(self._position[0], self._position[1], self._position[2]))

    def set_view_direction(self, x, y, opencl, propagation_length):
        rel_phi = x / self._width
        rel_theta = y / self._height
        self._view_direction[0] = (rel_phi - 0.5) * 2.0 * np.pi
        self._view_direction[1] = (-rel_theta + 0.5) * 0.8 * np.pi
        self.calculate_ray_origins(opencl)
        if not self.trace_on:
            self.surface_id(opencl, propagation_length)

    def rotate_walk_direction(self, delta, opencl):
        self._walk_direction += delta
        self.calculate_ray_origins(opencl)
        if not self.trace_on:
            self.surface_id(opencl, propagation_length)

    def get_position(self):
        return self._position

    def get_direction(self):
        return [self._view_direction[0] + self._walk_direction, self._view_direction[1]]

    def calculate_ray_origins(self, opencl):
        pos_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_position()))
        dir_in_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self.get_direction()))
        field_of_view_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self._field_of_view))
        camera_style_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(self._camera_style))
        ray_spacing_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.float32(self._ray_spacing))

        opencl.prg.calculate_ray_origins(
            opencl.queue,
            (self._width, self._height),
            None,
            self._width_g,
            self._height_g,
            pos_in_g,
            dir_in_g,
            field_of_view_g,
            ray_spacing_g,
            camera_style_g,
            self._pos_out_a.data,
            self._dir_out_a.data
        )

    def trace(self, opencl, propagation_length):
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=np.float32(propagation_length))

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

    def surface_id(self, opencl, propagation_length):
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=np.float32(propagation_length))

        opencl.prg.surface_id(
            opencl.queue,
            (self._width, self._height),
            None,
            self._width_g,
            self._height_g,
            self._pos_out_a.data,
            self._dir_out_a.data,
            propagation_length_g,
            self._env_dim_g,
            self._environment_g,
            self._surface_rendered_a.data,
            self._intensity_a.data
        )

    def render_surface(self, opencl, propagation_length, number_of_samples):
        propagation_length_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=np.float32(propagation_length))
        number_of_samples_g = cl.Buffer(opencl.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=np.int32(number_of_samples))

        opencl.prg.render_surface(
            opencl.queue,
            (self._env_dim[0], self._env_dim[1], self._env_dim[2]),
            None,
            self._seed_g,
            propagation_length_g,
            number_of_samples_g,
            self._env_dim_g,
            self._environment_g,
            self._surface_rendered_a.data
        )

    def fill_surface(self, surface):
        values = self.get_intensity()
        surfarray.blit_array(surface, values)

    def snapshot(self, opencl, field_of_view, propagation_length, nr_of_samples=100):
        # TODO: refactor with new output structure
        trace_out = np.zeros((self._width, self._height, 3), dtype=np.uint32)
        for sample in range(nr_of_samples):
            self.trace(opencl, propagation_length)
            trace_out += self.get_intensity()

        surface_id_out = np.zeros((self._width, self._height, 3), dtype=np.uint32)
        self.surface_id(opencl, propagation_length)
        surface_id_out += self.get_intensity()
        surface_id_out *= nr_of_samples

        imsave(
            'snapshot.png',
            np.concatenate(
                [
                    trace_out.transpose((1, 0, 2)),
                    surface_id_out.transpose((1, 0, 2)),
                ],
                axis=0,
            )
        )

    def get_intensity(self):
        return self._intensity_a.get()

    def get_distance(self):
        return self._distance_a.get()


def hash_fun(x):
    y = x[0] << 24 + x[1] << 16 + x[2] << 8 + x[3] * 5
    return y