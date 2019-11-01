import numpy as np
from imageio import imread, imsave
from time import time
import pygame
from pygame import Surface
import sys

from camera import Camera
from opencl_handler import OpenCLHandler
from environment import Environment, prison, street


def handle_events(opencl, camera, environment, propagation_length):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            camera.set_view_direction(x, y, opencl, propagation_length)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            environment.build(camera.get_position())
            camera.update_environment_buffer(opencl, environment)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            # if event.key == pygame.K_LEFT:
            #     camera.rotate_walk_direction(-0.1*np.pi, opencl)
            # if event.key == pygame.K_RIGHT:
            #     camera.rotate_walk_direction(0.1*np.pi, opencl)
            if event.key == pygame.K_DOWN:
                camera.move(-1, environment, opencl)
            if event.key == pygame.K_UP:
                camera.move(1, environment, opencl)
            if event.key == pygame.K_SPACE:
                camera.trace_on = not camera.trace_on
            if event.key == pygame.K_s:
                camera.snapshot(opencl, field_of_view=0.5*np.pi, propagation_length=0.99, nr_of_samples=100)

def main():
    env_dim = [40, 40, 40]
    width = 720
    height = 450
    propagation_length = 0.99
    number_of_samples = 10
    position = [2, env_dim[1] - 2, 5]
    view_direction = [0, 0]
    walk_direction = 0

    environment = Environment(env_dim)
    environment.load(prison, env_dim)

    opencl = OpenCLHandler(
        [
            'opencl/main.cl',
            'opencl/propagate.cl',
            'opencl/origins.cl',
            'opencl/surfaceid.cl',
            'opencl/trace.cl',
            'opencl/render_surface.cl',
        ],
        device=1,
    )
    camera = Camera(position, view_direction, walk_direction, width, height, opencl, environment)
    camera.calculate_ray_origins(opencl)

    print('rendering surface colors...')
    camera.render_surface(opencl, propagation_length, number_of_samples)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    #screen = pygame.display.set_mode((width, height), (pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF))
    clock = pygame.time.Clock()

    i = 0

    while True:
        #environment.run()
        handle_events(opencl, camera, environment, propagation_length)
        if camera.trace_on:
            camera.trace(opencl, propagation_length)
        camera.fill_surface(screen)
        pygame.display.flip()
        clock.tick()

        i += 1
        if i % 10 == 0:
            print('fps: {:2.2f}'.format(clock.get_fps()))

if __name__ == '__main__':
    main()