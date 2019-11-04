import numpy as np

OPEN_SPACE = 0
WALL         = 0x0100000000000000
LOCAL_LIGHT  = 0x0200000000000000
GLOBAL_LIGHT = 0x0300000000000000
MIRROR       = 0x0400000000000000
MIRROR_0XFF  = 0x00FF000000000000
MIRROR_0X7F  = 0x007F000000000000
MIRROR_0X3F  = 0x003F000000000000

RED   = 0xFF000000
GREEN = 0x00FF0000
BLUE  = 0x0000FF00

DARK_RED = 0x7F000000
DARK_GREEN = 0x007F0000
DARK_BLUE = 0x00007F00

WHITE = 0xFFFFFF00
GREY = 0x7F7F7F00
DARK_GREY = 0x3F3F3F00

PURPLE = 0xFF00FF00
YELLOW = 0xFFFF0000
CYAN = 0x00FFFF00

BLACK = 0


class Aggregate(object):
    def __init__(self, dimension, position):
        self._position = position
        self._array = np.array(dimension)

    def export(self, target_array):
        target_array[self._position] = self._array


class Environment(object):
    def __init__(self, dimensions):
        self._dimensions = dimensions
        self._env_array = self._default_array()

        self._aggregates = []
        self.current_build_option = (WALL | DARK_GREY)

    def load(self, fun, env_dim):
        self._dimensions = env_dim
        self._env_array = fun(env_dim)

    def _default_array(self):
        return np.zeros(self._dimensions, dtype=np.uint64)

    def export(self):
        array = self._default_array()

        for agg in self._aggregates:
            agg.export(array)

    def build(self, position):
        x = int(position[0])
        y = int(position[1])
        z = int(position[2])

        self._env_array[x, y, z] = self.current_build_option

        print('building at {}, {}, {}'.format(x, y, z))

    def run(self):
        '''
        "run" environment, i.e. move stuff around, flip blocks off/on etc
        '''
        random = np.random.randint(0, 2, self._dimensions, dtype=np.uint64) << 56
        self._env_array = (self._env_array & 0x00FFFFFFFFFFFFFF) + ((~(self._env_array >> 56) & random) << 56)
        pass

def prison(d):
    array = np.zeros((d[0], d[1], d[2]), dtype=np.uint64) + (WALL | DARK_BLUE | RED)
    array[1:d[0] - 1, 1:d[1] - 1, 1:d[2]] = OPEN_SPACE
    array[1:d[0] - 1, 1:d[1] - 1, 0] = (WALL | GREY )#| MIRROR_0X3F)
    array[1:d[0] - 1, 1:10, d[2]//3] = (WALL | GREY)
    array[0, 1:d[1] - 1, 1:d[2]] = (WALL | RED )#| MIRROR_0X3F)
    array[d[0] - 1, 1:d[1] - 1, 1:d[2]] = (WALL | RED )#| MIRROR_0X3F)
    array[d[0] - 2, d[1] - 2, 1:d[2]] = (LOCAL_LIGHT | WHITE)

    array[3:d[0] - 3:4, 3:d[1] - 3:4, 3:d[2] - 3:4] = (WALL | YELLOW)

    return array

def street(d):
    array = np.zeros((d[0], d[1], d[2]), dtype=np.uint64)  # black wall
    array[1:d[0] - 1, 1:d[1] - 1, 1:d[1] - 1] = OPEN_SPACE  # open space
    array[1:d[0] - 1, 1:d[1] - 1, 0] = (WALL | GREY)  # grey floor
    #array[d // 2 - 7: d // 2 + 7, d // 2 - 7: d // 2 + 7, d // 2 - 7: d // 2 + 7] = SMOKE
    array[d[0] // 2 - 5: d[0] // 2 + 5, d[1] // 2 - 5: d[1] // 2 + 5, d[2] // 2 - 5: d[2] // 2 + 5] = (LOCAL_LIGHT | RED | GREEN | BLUE)  # white block in the middle

    return array
