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

        # We need to keep track of what we add to the space to be able to remove it later
        self._static_shapes = [] # Walls, floor - part of the level boundary
        self._dynamic_shapes = [] # Player-placed walls, boxes, etc.
        self._goal_zones = []

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
        # Player-placed walls are treated as "dynamic" from a level design perspective
        self._dynamic_shapes.append(segment)

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
        self._goal_zones.append(shape)
        return shape

    def clear_dynamic_objects(self):
        """Removes all player-placed objects from the space."""
        for shape in self._dynamic_shapes:
            self.space.remove(shape, shape.body)
        self._dynamic_shapes.clear()

        for shape in self._goal_zones:
            self.space.remove(shape, shape.body)
        self._goal_zones.clear()

    def get_serializable_data(self):
        """Returns a list of dictionaries representing the dynamic objects."""
        data = []
        for shape in self._dynamic_shapes:
            if isinstance(shape, pymunk.Segment):
                data.append({
                    'type': 'wall',
                    'start': tuple(shape.a),
                    'end': tuple(shape.b)
                })
            elif isinstance(shape, pymunk.Poly): # This assumes only boxes are dynamic polys
                width = shape.get_vertices()[1].x - shape.get_vertices()[0].x
                height = shape.get_vertices()[2].y - shape.get_vertices()[1].y
                data.append({
                    'type': 'box',
                    'position': tuple(shape.body.position),
                    'size': (width, height)
                })
        for shape in self._goal_zones:
             if isinstance(shape, pymunk.Poly):
                # This assumes rectangular polys created with create_box
                width = shape.get_vertices()[1].x - shape.get_vertices()[0].x
                height = shape.get_vertices()[2].y - shape.get_vertices()[1].y
                data.append({
                    'type': 'goal',
                    'position': tuple(shape.body.position),
                    'size': (width, height)
                })
        return data

    def load_from_data(self, data):
        """Clears the current dynamic objects and loads new ones from data."""
        self.clear_dynamic_objects()
        for obj_data in data:
            if obj_data['type'] == 'wall':
                self.add_static_segment(obj_data['start'], obj_data['end'])
            elif obj_data['type'] == 'goal':
                self.add_goal_zone(obj_data['position'], obj_data['size'])
            elif obj_data['type'] == 'box':
                self.add_dynamic_box(obj_data['position'], obj_data['size'])

    def add_dynamic_box(self, position, size=(40, 40), mass=1.0):
        """Adds a dynamic, pushable box to the space."""
        body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
        body.position = position
        shape = pymunk.Poly.create_box(body, size)
        shape.friction = 0.7
        shape.elasticity = 0.4
        self.space.add(body, shape)
        self._dynamic_shapes.append(shape)
        return shape
