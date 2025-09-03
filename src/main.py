# src/main.py

import pygame
import pymunk.pygame_util
import torch
import pygame_gui

# Import our local modules
import config
from environment import Environment
from agent import Agent
from ai_model import DQNAgent

def main():
    """
    Main function to run the application.
    """
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Py-AI Warehouse")
    clock = pygame.time.Clock()

    # --- UI Manager Setup ---
    ui_manager = pygame_gui.UIManager((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    # --- Environment and Agent Setup ---
    env = Environment()
    # Add static boundaries to the environment
    env.add_static_segment((0, config.SCREEN_HEIGHT - 5), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - 5)) # Floor
    env.add_static_segment((5, 0), (5, config.SCREEN_HEIGHT)) # Left wall
    env.add_static_segment((config.SCREEN_WIDTH - 5, 0), (config.SCREEN_WIDTH - 5, config.SCREEN_HEIGHT)) # Right wall
    env.add_static_segment((0, 5), (config.SCREEN_WIDTH, 5)) # Ceiling

    agent = Agent(env.space, start_pos=(100, config.SCREEN_HEIGHT - 100))

    # The AI Agent is set up but not used until we are in simulation mode
    ai_agent = DQNAgent(agent.n_observations, agent.n_actions)

    # --- Goal and Collision Handling Setup ---
    goal_zone = env.add_goal_zone(position=(config.SCREEN_WIDTH - 100, config.SCREEN_HEIGHT / 2), size=(50, 100))

    def goal_reached_handler(arbiter, space, data):
        data["agent_reached_goal"] = True
        return True

    handler_data = {"agent_reached_goal": False}

    # The 'begin' function is called at the start of a collision. We pass our
    # mutable 'handler_data' dict to it using a lambda.
    env.space.on_collision(
        config.COLLISION_TYPE_AGENT,
        config.COLLISION_TYPE_GOAL,
        begin=lambda arbiter, space, data: goal_reached_handler(arbiter, space, handler_data)
    )

    # --- Application State ---
    current_mode = 'setup_mode' # Start in setup mode
    is_training = False
    is_paused = False
    episode_counter = 0
    selected_object_type = None # For the editor

    # --- UI Elements ---
    mode_switch_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (180, 40)),
                                                     text='Switch to Simulation',
                                                     manager=ui_manager)
    start_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((200, 10), (150, 40)),
                                                  text='Start Training',
                                                  manager=ui_manager)
    pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((360, 10), (100, 40)),
                                                 text='Pause',
                                                 manager=ui_manager)
    reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((470, 10), (100, 40)),
                                                 text='Reset',
                                                 manager=ui_manager)

    # --- Editor UI Elements (initially visible in setup mode) ---
    editor_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((config.SCREEN_WIDTH - 220, 60), (210, 300)),
                                               manager=ui_manager)

    wall_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (180, 40)),
                                                text='Select Wall',
                                                manager=ui_manager,
                                                container=editor_panel)

    goal_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 60), (180, 40)),
                                               text='Select Goal Zone',
                                               manager=ui_manager,
                                               container=editor_panel)

    # --- Main Application Loop ---
    running = True
    while running:
        time_delta = clock.tick(config.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Pass events to the UI manager
            ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == mode_switch_button:
                    if current_mode == 'setup_mode':
                        current_mode = 'simulation_mode'
                        mode_switch_button.set_text('Switch to Setup Mode')
                        editor_panel.hide()
                    else:
                        current_mode = 'setup_mode'
                        mode_switch_button.set_text('Switch to Simulation')
                        editor_panel.show()
                elif event.ui_element == start_button:
                    is_training = True
                    print("Starting training...")
                elif event.ui_element == pause_button:
                    is_paused = not is_paused
                    pause_button.set_text('Resume' if is_paused else 'Pause')
                    print("Training paused" if is_paused else "Training resumed.")
                elif event.ui_element == reset_button:
                    agent.reset()
                    handler_data["agent_reached_goal"] = False
                    episode_counter = 0
                    is_training = False
                    is_paused = False
                    pause_button.set_text('Pause')
                    print("Agent and training progress have been reset.")
                elif event.ui_element == wall_button:
                    selected_object_type = 'wall'
                    print("Selected object: Wall")
                elif event.ui_element == goal_button:
                    selected_object_type = 'goal'
                    print("Selected object: Goal Zone")

            # Handle object placement in setup mode
            if current_mode == 'setup_mode' and event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the click was on the main screen, not on a UI element
                if not ui_manager.get_focus_set():
                    if selected_object_type == 'wall':
                        # A simple horizontal wall segment
                        start_pos = (event.pos[0] - 40, event.pos[1])
                        end_pos = (event.pos[0] + 40, event.pos[1])
                        env.add_static_segment(start_pos, end_pos)
                        print(f"Placed Wall at {event.pos}")
                    elif selected_object_type == 'goal':
                        env.add_goal_zone(position=event.pos, size=(50, 100))
                        print(f"Placed Goal Zone at {event.pos}")


        # --- Mode-Specific Logic ---
        if current_mode == 'setup_mode':
            # In setup mode, we will handle object placement and UI interaction
            pass # Logic is now handled in the event loop
        elif current_mode == 'simulation_mode':
            if not is_paused:
                # Always step the physics simulation if we are in sim mode and not paused
                env.space.step(1.0 / config.FPS)

                if is_training:
                    # If training is active, perform one step of the learning process
                    state_tuple = agent.get_state()
                    state = torch.tensor(state_tuple, dtype=torch.float32, device=ai_agent.device).unsqueeze(0)

                    action = ai_agent.select_action(state)
                    agent.perform_action(action.item())

                    next_state_tuple = agent.get_state()
                    done = handler_data["agent_reached_goal"]

                    goal_reward = 100.0
                    living_penalty = -0.1
                    distance_reward_factor = 100.0
                    if done:
                        reward = goal_reward
                    else:
                        distance_reward = (next_state_tuple[0] - state_tuple[0]) * distance_reward_factor
                        reward = distance_reward + living_penalty

                    reward_tensor = torch.tensor([reward], device=ai_agent.device)
                    done_tensor = torch.tensor([done], device=ai_agent.device)
                    next_state = None if done else torch.tensor(next_state_tuple, dtype=torch.float32, device=ai_agent.device).unsqueeze(0)

                    ai_agent.memory.push(state, action, next_state, reward_tensor, done_tensor)
                    ai_agent.learn()

                    if done:
                        episode_counter += 1
                        print(f"Episode {episode_counter} finished.")
                        agent.reset()
                        handler_data["agent_reached_goal"] = False

        # Update the UI Manager
        ui_manager.update(time_delta)

        # --- Drawing ---
        screen.fill(config.COLOR_BACKGROUND)

        # Draw the physics space
        env.space.debug_draw(pymunk.pygame_util.DrawOptions(screen))

        # Draw the UI
        ui_manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
