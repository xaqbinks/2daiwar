# src/environment.py

import pymunk
import config

class Environment:
    """
    Manages the physics simulation space and its objects.
    """
    def __init__(self):
        """
        Initializes the physics space.
        """
        self.space = pymunk.Space()
        self.space.gravity = (0, 981) # Standard gravity in pixel-space (y-down)
        self._static_shapes = []

    def add_static_segment(self, start_pos, end_pos, radius=2):
        """
        Adds a static line segment to the physics space.
        These are immovable objects like floors and walls.

        :param start_pos: The starting (x, y) coordinate of the line.
        :param end_pos: The ending (x, y) coordinate of the line.
        :param radius: The thickness of the line.
        """
        segment = pymunk.Segment(self.space.static_body, start_pos, end_pos, radius)
        segment.elasticity = 0.8
        segment.friction = 0.9
        self.space.add(segment)
        self._static_shapes.append(segment)

    def get_static_shapes(self):
        """
        Returns the list of static shapes for rendering.
        """
        return self._static_shapes

    def add_goal_zone(self, position, size):
        """
        Adds a sensor shape that acts as the goal zone.
        """
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Poly.create_box(body, size)
        shape.sensor = True # Makes it a sensor so it doesn't cause collisions
        shape.collision_type = config.COLLISION_TYPE_GOAL
        shape.color = config.COLOR_GOAL # For debug drawing
        self.space.add(body, shape)
        return shape
