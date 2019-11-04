import numpy as np

OPEN_SPACE = 0
WALL = 1
LOCAL_LIGHT = 2
GLOBAL_LIGHT = 3
SMOKE = 4

RED   = 0xFF000000
GREEN = 0x00FF0000
BLUE  = 0x0000FF00

DARK_RED = 0x7F000000
DARK_GREEN = 0x007F0000
DARK_BLUE = 0x00007F00

WHITE = 0xFFFFFF00
GREY = 0x7F7F7F00

PURPLE = 0xFF00FF00
YELLOW = 0xFFFF0000
CYAN = 0x00FFFF00

# LIGHT_RED = 128
# LIGHT_GREEN = 32
# LIGHT_BLUE = 8
#
# DARK_RED = 64
# DARK_GREEN = 16
# DARK_BLUE = 4
#
# WHITE = RED + GREEN + BLUE
# LIGHT_GREY = LIGHT_RED + LIGHT_GREEN + LIGHT_BLUE
# DARK_GREY = DARK_RED + DARK_GREEN + DARK_BLUE
BLACK = 0

DEFAULT_DIMENSION = 50

class Environment(object):
    def __init__(self):
        self._env_dim = None
        self._env_array = None

    def load(self, fun, env_dim):
        self._env_dim = env_dim
        self._env_array = fun(env_dim)


def prison(d=DEFAULT_DIMENSION):
    array = np.zeros((d[0], d[1], d[2]), dtype=np.uint32) + (WALL | DARK_RED)
    array[1:d[0] - 1, 1:d[1] - 1, 1:d[2]] = OPEN_SPACE
    array[1:d[0] - 1, 1:d[1] - 1, 0] = (WALL | BLUE)
    array[0, 1:d[1] - 1, 1:d[2] - 1] = (WALL | GREEN)
    array[d[0] - 2, d[1] - 2, 1:d[2] - 1] = (LOCAL_LIGHT | WHITE)

    return array

def prison2(d):
    array = np.zeros((d[0], d[1], d[2]), dtype=np.uint64) + (WALL | DARK_BLUE | RED)
    array[1:d[0] - 1, 1:d[1] - 1, 1:d[2]] = OPEN_SPACE
    array[1:d[0] - 1, 1:d[1] - 1, 0] = (WALL | GREY )#| MIRROR_0X3F)
    array[1:d[0] - 1, 1:10, d[2]//3] = (WALL | GREY)
    array[0, 1:d[1] - 1, 1:d[2]] = (WALL | RED )#| MIRROR_0X3F)
    array[d[0] - 1, 1:d[1] - 1, 1:d[2]] = (WALL | RED )#| MIRROR_0X3F)
    array[d[0] - 2, d[1] - 2, 1:d[2]] = (LOCAL_LIGHT | WHITE)

    array[3:d[0] - 3:4, 3:d[1] - 3:4, 3:d[2] - 3:4] = (WALL | YELLOW)

    return array

def street(d=DEFAULT_DIMENSION):
    array = np.zeros((d[0], d[1], d[2]), dtype=np.uint32)  # black wall
    array[1:d[0] - 1, 1:d[1] - 1, 1:d[1] - 1] = OPEN_SPACE  # open space
    array[1:d[0] - 1, 1:d[1] - 1, 0] = (WALL | GREY)  # grey floor
    #array[d // 2 - 7: d // 2 + 7, d // 2 - 7: d // 2 + 7, d // 2 - 7: d // 2 + 7] = SMOKE
    array[d[0] // 2 - 5: d[0] // 2 + 5, d[1] // 2 - 5: d[1] // 2 + 5, d[2] // 2 - 5: d[2] // 2 + 5] = (LOCAL_LIGHT | RED | GREEN | BLUE)  # white block in the middle

    return array
