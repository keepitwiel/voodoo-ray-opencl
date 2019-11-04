import numpy as np
from imageio import imread, imsave
from time import time
import pygame
from pygame import Surface
import sys

from early.camera import Camera
from early.opencl_handler import OpenCLHandler
from early.environment import Environment, prison, prison2, street


def handle_events(opencl, camera, environment):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            camera.set_view_direction(x, y)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                sys.exit()
            if event.key == pygame.K_LEFT:
                camera.rotate_walk_direction(-0.1*np.pi)
            if event.key == pygame.K_RIGHT:
                camera.rotate_walk_direction(0.1*np.pi)
            if event.key == pygame.K_DOWN:
                camera.move(-1, environment)
            if event.key == pygame.K_UP:
                camera.move(1, environment)
            if event.key == pygame.K_s:
                camera.switch()
            if event.key == pygame.K_SPACE:
                camera.snapshot(opencl, field_of_view=0.5*np.pi, propagation_length=0.1, nr_of_samples=100)

def main():
    env_dim = [20, 20, 100]
    width = 640
    height = 400
    field_of_view = 0.5 * np.pi
    propagation_length = 0.99
    position = [2, 2, env_dim[2]//2]
    view_direction = [0, 0]
    walk_direction = 0

    environment = Environment()
    environment.load(prison2, env_dim)

    opencl = OpenCLHandler('voodooray.cl', 1)
    camera = Camera(position, view_direction, walk_direction, width, height, opencl, environment)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    #screen = pygame.display.set_mode((width, height), (pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF))
    clock = pygame.time.Clock()

    i = 0

    while True:
        t0 = time()
        handle_events(opencl, camera, environment)

        t1 = time()
        if camera._trace:
            camera.trace(opencl, field_of_view, propagation_length)
        else:
            camera.lidar(opencl, field_of_view, propagation_length)

        t2 = time()
        camera.fill_surface(screen)

        t3 = time()
        pygame.display.flip()

        t4 = time()
        clock.tick()

        i += 1
        if i % 10 == 0:
            print('times: {0:2.4f}, {1:2.4f}, {2:2.4f}, {3:2.4f}, (total: {4:2.4f}, fps: {5:2.2f})'.format(
                t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0, clock.get_fps()))

if __name__ == '__main__':
    main()