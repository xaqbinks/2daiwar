# Py-AI Warehouse

## 1. Overview

Py-AI Warehouse is a 2D sandbox environment where you can act as a researcher, setting up challenges to train AI agents using genuine reinforcement learning. The core innovation is the ability to "save" an agent's learned skills (as "Skill Genes") and "implant" them into new agents, creating a strategic meta-game of building increasingly capable AI.

This project was built based on a Game Design Document, implementing all 6 initial milestones and a final polishing phase.

## 2. Features

*   **2D Physics Sandbox:** Built with Pygame and the Pymunk physics library.
*   **Reinforcement Learning AI:** The agent is powered by a Deep Q-Network (DQN) implemented in PyTorch.
*   **Two Main Modes:**
    *   **Setup Mode:** Design and build your own levels and challenges.
    *   **Simulation Mode:** Watch the AI agent attempt to solve the challenge you've built.
*   **Environment Editor:** In Setup Mode, you can place, move, and configure various objects:
    *   Static Walls
    *   Dynamic, Pushable Boxes
    *   A Goal Zone for the agent to reach
*   **Live Simulation Controls:** Start, pause, and reset the AI training simulation using UI buttons.
*   **The "Marketplace of Minds":**
    *   **Save Skills:** Save a trained agent's neural network weights as a `.pth` file (a "Skill Gene").
    *   **Load Skills:** Load a previously saved skill into an agent to give it a "head start" on a new challenge (Transfer Learning).
    *   **Gene Library:** A UI panel allows you to manage your saved skills.
*   **Scene Management:** Save and load your custom-built environment layouts as `.json` files.
*   **Challenge Rooms:** The application comes with several pre-built challenge rooms to get you started.

## 3. Setup and Installation

### Dependencies

All required Python libraries are listed in `requirements.txt`. To install them, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## 4. How to Run

To run the application, execute the `main.py` script from the root of the project:

```bash
python3 src/main.py
```

## 5. How to Use

The application will start in **Setup Mode**.

### Setup Mode

This is where you build your level.
*   **Object Palette (Right Panel):** Select an object type to place (Wall, Goal Zone, Box).
*   **Placing Objects:** After selecting an object, click anywhere on the main screen to place it.
*   **Challenge Rooms (Left Panel):** You can load pre-made levels from this list.
*   **Scene Management (Editor Panel):**
    *   `Clear Scene`: Removes all objects you've placed.
    *   `Save Scene`: Opens a file dialog to save your current layout as a `.json` file in the `scenes/` directory.
    *   `Load Scene`: Opens a file dialog to load a `.json` layout from the `scenes/` directory.
*   **Gene Library (Bottom-Right Panel):**
    *   `Save Skill`: Opens a window to enter a name and save the current AI's brain.
    *   `Load Skill`: Loads a saved brain into the current AI.
    *   `Refresh List`: Updates the list of saved skills from the `saved_models/` directory.

### Simulation Mode

This is where the AI learns.
*   Click the **"Switch to Simulation"** button to enter this mode. All editor panels will hide.
*   **Live Stats (Top-Right):** You can monitor the current Episode and the AI's Epsilon (exploration rate).
*   **Start Training:** Begins the AI's learning process. The AI will take control of the agent and try to solve the level.
*   **Pause/Resume:** Pauses the physics and learning.
*   **Reset:** Resets the agent to its starting position and resets all training progress.

### A Typical Workflow

1.  Start the app (you are in Setup Mode).
2.  Load a challenge from the "Challenge Rooms" panel or build your own level.
3.  Click "Switch to Simulation".
4.  Click "Start Training".
5.  Watch the agent and the live stats to see its progress.
6.  Once the agent has learned the task, click "Pause".
7.  Click "Switch to Setup Mode".
8.  In the "Gene Library" panel, click "Save Skill" and give it a name.
9.  Click "Clear Scene", then build a new, harder level.
10. Select your newly saved skill from the list and click "Load Skill".
11. Switch back to Simulation Mode and start training. The agent should now learn the new, harder task much more quickly.
