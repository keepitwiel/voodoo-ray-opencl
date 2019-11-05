import numpy as np
import pygame
import sys

from camera import Camera, TRACE, LIDAR
from opencl_handler import OpenCLHandler
from environment import Environment, prison


def handle_events(opencl, camera, environment, field_of_view, propagation_length):
    rerender = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            camera.set_view_direction(x, y)
            rerender = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            environment.build(camera.get_position())
            camera.update_environment_buffer(opencl, environment)
            rerender = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            if event.key == pygame.K_DOWN:
                camera.move(-1, environment)
                rerender = True
            if event.key == pygame.K_UP:
                camera.move(1, environment)
                rerender = True
            if event.key == pygame.K_SPACE:
                camera.switch()
            if event.key == pygame.K_s:
                camera.snapshot(opencl, propagation_length=0.1)

    if rerender:
        camera.lidar(opencl, field_of_view, propagation_length)
        camera.trace(opencl, propagation_length)
        camera.distance_blur(opencl)
        pass

def main():
    env_dim = [40, 40, 40]
    width = 640
    height = 400
    propagation_length = 0.99
    position = [2, env_dim[1] - 2, 5]
    view_direction = [0, 0]
    walk_direction = 0
    field_of_view = 0.2 * np.pi

    environment = Environment(env_dim)
    environment.load(prison, env_dim)

    opencl = OpenCLHandler(
        [
            'voodooray.cl',
        ],
        device=1,
    )
    camera = Camera(position, view_direction, walk_direction, width, height, opencl, environment)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    #screen = pygame.display.set_mode((width, height), (pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF))
    clock = pygame.time.Clock()

    i = 0

    while True:
        handle_events(opencl, camera, environment, field_of_view, propagation_length)
        camera.fill_surface(screen)
        pygame.display.flip()
        clock.tick()

        i += 1
        if i % 10 == 0:
            print('fps: {:2.2f}'.format(clock.get_fps()))


if __name__ == '__main__':
    main()
