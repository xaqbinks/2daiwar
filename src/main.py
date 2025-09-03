# src/main.py

import pygame
import pymunk
import pymunk.pygame_util

# Import our local modules
import config
from environment import Environment
from agent import Agent

def main():
    """
    Main function to run the game.
    """
    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Py-AI Warehouse")
    clock = pygame.time.Clock()

    # Pymunk debugging draw options
    # This utility handles rendering Pymunk shapes in Pygame
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Create the environment
    env = Environment()

    # Create the agent
    agent = Agent(env.space, start_pos=(100, config.SCREEN_HEIGHT - 100))

    # Add static boundaries to the environment
    # Floor
    env.add_static_segment((0, config.SCREEN_HEIGHT - 5), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - 5))
    # Left wall
    env.add_static_segment((5, 0), (5, config.SCREEN_HEIGHT))
    # Right wall
    env.add_static_segment((config.SCREEN_WIDTH - 5, 0), (config.SCREEN_WIDTH - 5, config.SCREEN_HEIGHT))
    # Ceiling
    env.add_static_segment((0, 5), (config.SCREEN_WIDTH, 5))


    # Main game loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    agent.jump()

        # Handle continuous key presses for movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            agent.move(-1)
        if keys[pygame.K_RIGHT]:
            agent.move(1)

        # Clear the screen with the background color
        screen.fill(config.COLOR_BACKGROUND)

        # Draw the physics space
        # The debug draw function is perfect for visualizing the simulation
        env.space.debug_draw(draw_options)

        # Update the physics simulation
        # Step forward in time by a fixed amount
        dt = 1.0 / config.FPS
        env.space.step(dt)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(config.FPS)

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()
