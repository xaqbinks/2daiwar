# src/main.py

import pygame
import pymunk.pygame_util
import torch
import pygame_gui
import os
import time
import json
import math

# Import our local modules
import config
from environment import Environment
from agent import Agent
from ai_model import DQNAgent

def scan_for_saved_models():
    """Scans the saved_models directory for .pth files."""
    if not os.path.exists('saved_models'):
        return []
    return [f for f in os.listdir('saved_models') if f.endswith('.pth')]

def scan_for_challenges():
    """Scans the challenges directory for .json files."""
    if not os.path.exists('challenges'):
        return []
    return [f for f in os.listdir('challenges') if f.endswith('.json')]

def show_message_box(manager, title, message):
    """Helper function to show a message window."""
    pygame_gui.windows.UIMessageWindow(
        rect=pygame.Rect((config.SCREEN_WIDTH / 2 - 150, config.SCREEN_HEIGHT / 2 - 100), (300, 200)),
        html_message=message,
        manager=manager,
        window_title=title
    )

def main():
    """
    Main function to run the application.
    """
    # --- Ensure directories exist ---
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("scenes", exist_ok=True)
    os.makedirs("challenges", exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Py-AI Warehouse")
    clock = pygame.time.Clock()

    # --- UI Manager Setup ---
    ui_manager = pygame_gui.UIManager((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    # --- Environment and Agent Setup ---
    env = Environment()
    # Add static boundaries to the environment directly, not as part of the dynamic level
    static_boundaries = [
        pymunk.Segment(env.space.static_body, (0, config.SCREEN_HEIGHT - 5), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - 5), 5),
        pymunk.Segment(env.space.static_body, (5, 0), (5, config.SCREEN_HEIGHT), 5),
        pymunk.Segment(env.space.static_body, (config.SCREEN_WIDTH - 5, 0), (config.SCREEN_WIDTH - 5, config.SCREEN_HEIGHT), 5),
        pymunk.Segment(env.space.static_body, (0, 5), (config.SCREEN_WIDTH, 5), 5)
    ]
    for line in static_boundaries:
        line.elasticity = 0.8
        line.friction = 0.9
        line.color = config.COLOR_STATIC
    env.space.add(*static_boundaries)

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
    save_window = None

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

    # --- Stats UI Elements ---
    stats_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((580, 10), (250, 100)),
                                              manager=ui_manager)

    episode_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 10), (220, 20)),
                                                text="Episode: 0", manager=ui_manager, container=stats_panel)
    epsilon_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 40), (220, 20)),
                                                text="Epsilon: 0.00", manager=ui_manager, container=stats_panel)

    # --- Editor UI Elements (initially visible in setup mode) ---
    editor_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((config.SCREEN_WIDTH - 220, 60), (210, 350)),
                                               manager=ui_manager)

    wall_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (180, 40)),
                                                text='Select Wall',
                                                manager=ui_manager,
                                                container=editor_panel)

    goal_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 60), (180, 40)),
                                               text='Select Goal Zone',
                                               manager=ui_manager,
                                               container=editor_panel)

    clear_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 110), (180, 40)),
                                                text='Clear Scene',
                                                manager=ui_manager,
                                                container=editor_panel)

    save_scene_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 160), (180, 40)),
                                                     text='Save Scene',
                                                     manager=ui_manager,
                                                     container=editor_panel)

    load_scene_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 210), (180, 40)),
                                                     text='Load Scene',
                                                     manager=ui_manager,
                                                     container=editor_panel)

    box_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 260), (180, 40)),
                                              text='Select Box',
                                              manager=ui_manager,
                                              container=editor_panel)

    # --- Gene Library UI Elements (initially visible in setup mode) ---
    gene_library_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((config.SCREEN_WIDTH - 220, 370), (210, 340)),
                                                     manager=ui_manager)

    # --- Challenge Loader UI Elements ---
    challenge_panel = pygame_gui.elements.UIPanel(relative_rect=pygame.Rect((config.SCREEN_WIDTH - 440, 60), (210, 250)),
                                                   manager=ui_manager)

    challenge_list = pygame_gui.elements.UISelectionList(relative_rect=pygame.Rect((10, 10), (180, 150)),
                                                         item_list=[],
                                                         manager=ui_manager,
                                                         container=challenge_panel)

    load_challenge_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 170), (180, 40)),
                                                         text='Load Challenge',
                                                         manager=ui_manager,
                                                         container=challenge_panel)

    save_skill_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (180, 40)),
                                                      text='Save Skill',
                                                      manager=ui_manager,
                                                      container=gene_library_panel)

    load_skill_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 60), (180, 40)),
                                                      text='Load Skill',
                                                      manager=ui_manager,
                                                      container=gene_library_panel)

    skill_list = pygame_gui.elements.UISelectionList(relative_rect=pygame.Rect((10, 110), (180, 150)),
                                                     item_list=[], # Will be populated by scanning the directory
                                                     manager=ui_manager,
                                                     container=gene_library_panel)

    refresh_skills_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 270), (180, 40)),
                                                         text='Refresh List',
                                                         manager=ui_manager,
                                                         container=gene_library_panel)

    # --- Initial Population of UI Lists ---
    skill_list.set_item_list(scan_for_saved_models())
    challenge_list.set_item_list(scan_for_challenges())

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
                        gene_library_panel.hide()
                        challenge_panel.hide()
                    else:
                        current_mode = 'setup_mode'
                        mode_switch_button.set_text('Switch to Simulation')
                        editor_panel.show()
                        gene_library_panel.show()
                        challenge_panel.show()
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
                elif event.ui_element == box_button:
                    selected_object_type = 'box'
                    print("Selected object: Box")
                elif event.ui_element == clear_button:
                    env.clear_dynamic_objects()
                    show_message_box(ui_manager, "Success", "Scene cleared.")
                elif event.ui_element == save_scene_button:
                    file_dialog = pygame_gui.windows.UIFileDialog(
                        rect=pygame.Rect((config.SCREEN_WIDTH / 2 - 200, config.SCREEN_HEIGHT / 2 - 150), (400, 300)),
                        manager=ui_manager, window_title="Save Scene...", initial_file_path="scenes/",
                        allow_existing_files_only=False)
                elif event.ui_element == load_scene_button:
                    file_dialog = pygame_gui.windows.UIFileDialog(
                        rect=pygame.Rect((config.SCREEN_WIDTH / 2 - 200, config.SCREEN_HEIGHT / 2 - 150), (400, 300)),
                        manager=ui_manager, window_title="Load Scene...", initial_file_path="scenes/",
                        allow_existing_files_only=True)
                elif event.ui_element == refresh_skills_button:
                    skill_list.set_item_list(scan_for_saved_models())
                elif event.ui_element == load_skill_button:
                    selection = skill_list.get_single_selection()
                    if selection:
                        filepath = os.path.join('saved_models', selection)
                        ai_agent.load_model(filepath)
                        show_message_box(ui_manager, "Success", f"Skill '{selection}' loaded.")
                elif event.ui_element == save_skill_button:
                    if save_window is None:
                        save_window = pygame_gui.windows.UIWindow(
                            rect=pygame.Rect((config.SCREEN_WIDTH / 2 - 150, config.SCREEN_HEIGHT / 2 - 100), (300, 200)),
                            manager=ui_manager, window_display_title="Save Skill As...")

                        text_entry = pygame_gui.elements.UITextEntryLine(
                            relative_rect=pygame.Rect((10, 10), (260, 40)), manager=ui_manager, container=save_window)

                        confirm_save_button = pygame_gui.elements.UIButton(
                            relative_rect=pygame.Rect((10, 60), (100, 40)), text="Confirm", manager=ui_manager,
                            container=save_window, object_id="#confirm_save_button")
                elif event.ui_element == load_challenge_button:
                    selection = challenge_list.get_single_selection()
                    if selection:
                        filepath = os.path.join('challenges', selection)
                        with open(filepath, 'r') as f:
                            scene_data = json.load(f)
                        env.load_from_data(scene_data)
                        show_message_box(ui_manager, "Success", f"Challenge '{selection}' loaded.")

            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_object_id == 'window.#confirm_save_button':
                text_entry = [c for c in save_window.get_container().elements if isinstance(c, pygame_gui.elements.UITextEntryLine)][0]
                filename = text_entry.get_text()
                if filename:
                    if not filename.endswith(".pth"):
                        filename += ".pth"
                    filepath = os.path.join('saved_models', filename)
                    ai_agent.save_model(filepath)
                    skill_list.set_item_list(scan_for_saved_models())
                    show_message_box(ui_manager, "Success", f"Skill '{filename}' saved.")
                    save_window.kill()
                    save_window = None

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if save_window is not None and event.ui_element == save_window:
                    save_window = None

            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_CHOSEN:
                if "Save Scene..." in event.ui_element.window_title:
                    filepath = event.text
                    if not filepath.endswith(".json"):
                        filepath += ".json"
                    scene_data = env.get_serializable_data()
                    with open(filepath, 'w') as f:
                        json.dump(scene_data, f, indent=4)
                    show_message_box(ui_manager, "Success", "Scene saved.")
                elif "Load Scene..." in event.ui_element.window_title:
                    filepath = event.text
                    with open(filepath, 'r') as f:
                        scene_data = json.load(f)
                    env.load_from_data(scene_data)
                    show_message_box(ui_manager, "Success", "Scene loaded.")

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
                    elif selected_object_type == 'box':
                        env.add_dynamic_box(position=event.pos)
                        print(f"Placed Box at {event.pos}")


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

                    # Update Stats UI
                    eps_threshold = ai_agent.eps_end + (ai_agent.eps_start - ai_agent.eps_end) * \
                        math.exp(-1. * ai_agent.steps_done / ai_agent.eps_decay)
                    episode_label.set_text(f"Episode: {episode_counter}")
                    epsilon_label.set_text(f"Epsilon: {eps_threshold:.3f}")

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
