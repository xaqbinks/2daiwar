# src/agent.py

import pymunk
import config

class Agent:
    """
    The agent that will learn and interact with the environment.
    """
    def __init__(self, space, start_pos=(100, 100)):
        """
        Initializes the agent's physical body.

        :param space: The Pymunk space to add the agent to.
        :param start_pos: The starting (x, y) coordinate for the agent's torso.
        """
        self.space = space

        # --- Create the agent's torso ---
        torso_mass = 10
        torso_size = (30, 50)
        # Calculate moment of inertia for the torso
        torso_moment = pymunk.moment_for_box(torso_mass, torso_size)
        # Create the torso body
        self.torso_body = pymunk.Body(torso_mass, torso_moment)
        self.torso_body.position = start_pos
        # Create the torso shape
        torso_shape = pymunk.Poly.create_box(self.torso_body, torso_size)
        torso_shape.friction = 0.8
        torso_shape.elasticity = 0.1
        torso_shape.collision_type = config.COLLISION_TYPE_AGENT

        # --- Create the agent's head ---
        head_mass = 2
        head_radius = 12
        # Calculate moment of inertia for the head
        head_moment = pymunk.moment_for_circle(head_mass, 0, head_radius)
        # Create the head body
        self.head_body = pymunk.Body(head_mass, head_moment)
        # Position the head relative to the torso's starting position
        self.head_body.position = (start_pos[0], start_pos[1] - ((torso_size[1] / 2) + head_radius))
        # Create the head shape
        head_shape = pymunk.Circle(self.head_body, head_radius)
        head_shape.friction = 0.6
        head_shape.elasticity = 0.1
        head_shape.collision_type = config.COLLISION_TYPE_AGENT

        # --- Connect the body parts ---
        # Connect the head to the torso with a pivot joint.
        # The anchor points are relative to the center of each body.
        # The head is attached to the top-center of the torso.
        head_joint_anchor_on_torso = (0, 25)
        head_joint_anchor_on_head = (0, 0)
        head_joint = pymunk.PivotJoint(self.torso_body, self.head_body, head_joint_anchor_on_torso, head_joint_anchor_on_head)

        # Prevent the connected body parts from colliding with each other
        head_joint.collide_bodies = False

        # Add all the physics objects to the space
        self.space.add(self.torso_body, torso_shape, self.head_body, head_shape, head_joint)

        # Store the parts and joints for easy access
        self.body_parts = [self.torso_body, self.head_body]
        self.joints = [head_joint]

        # --- For state and action ---
        self.torso_size = torso_size
        self.head_radius = head_radius
        self.start_pos = start_pos
        self.action_space = {
            0: "left",
            1: "right",
            2: "jump",
            3: "none"
        }
        self.n_actions = len(self.action_space)
        self.n_observations = 4 # x_pos, y_pos, x_vel, y_vel

    def get_state(self):
        """
        Gets the current state of the agent.
        State is normalized for better NN performance.
        """
        pos_x = self.torso_body.position.x / config.SCREEN_WIDTH
        pos_y = self.torso_body.position.y / config.SCREEN_HEIGHT
        vel_x = self.torso_body.velocity.x / 1000
        vel_y = self.torso_body.velocity.y / 1000
        return (pos_x, pos_y, vel_x, vel_y)

    def perform_action(self, action_index):
        """
        Performs an action based on the action index from the AI.
        """
        action = self.action_space[action_index]
        if action == "left":
            self.move(-1)
        elif action == "right":
            self.move(1)
        elif action == "jump":
            self.jump()

    def reset(self):
        """
        Resets the agent to its initial state.
        """
        self.torso_body.position = self.start_pos
        self.torso_body.velocity = (0, 0)
        self.torso_body.angular_velocity = 0

        self.head_body.position = (self.start_pos[0], self.start_pos[1] - ((self.torso_size[1] / 2) + self.head_radius))
        self.head_body.velocity = (0, 0)
        self.head_body.angular_velocity = 0

    def move(self, direction):
        """
        Applies a horizontal force to the agent's torso to move it left or right.

        :param direction: An integer, -1 for left, 1 for right.
        """
        # Apply a force to the center of the torso
        force = (4000 * direction, 0)
        self.torso_body.apply_force_at_local_point(force, (0, 0))

    def jump(self):
        """
        Applies a vertical impulse to the agent's torso to make it jump.
        """
        # A simple check to see if the agent is on the ground before allowing a jump.
        # This can be made more sophisticated later (e.g., using raycasts).
        if abs(self.torso_body.velocity.y) < 1:
            impulse = (0, -8000) # Negative impulse is upwards
            self.torso_body.apply_impulse_at_local_point(impulse, (0, 0))
