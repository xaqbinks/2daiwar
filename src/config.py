# src/config.py

# Screen dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Frame rate
FPS = 60

# Colors (R, G, B)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BACKGROUND = (217, 217, 217) # A light grey
COLOR_STATIC = (45, 45, 45)       # A dark grey for walls/floor
COLOR_GOAL = (0, 255, 0, 100)     # Transparent green for goal zone

# --- Physics Collision Types ---
# Used to identify different kinds of shapes in collision handlers
COLLISION_TYPE_AGENT = 1
COLLISION_TYPE_GOAL = 2
