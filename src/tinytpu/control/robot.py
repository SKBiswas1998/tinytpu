"""Robot controller interface."""

class RobotController:
    """Motor control interface for differential drive robots."""

    def __init__(self, left_pin=None, right_pin=None, max_speed=0.3):
        self.max_speed = max_speed
        self._left_pin = left_pin
        self._right_pin = right_pin
        self._gpio = None

    def send(self, linear_x: float, angular_z: float):
        """Send velocity command to motors."""
        left = linear_x - angular_z * 0.5
        right = linear_x + angular_z * 0.5
        left = max(-self.max_speed, min(self.max_speed, left))
        right = max(-self.max_speed, min(self.max_speed, right))
        if self._gpio:
            self._set_motors(left, right)

    def stop(self):
        self.send(0, 0)

    def _set_motors(self, left, right):
        pass
