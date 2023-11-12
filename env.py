import gym
from gym import spaces
import numpy as np

class Soccer(gym.Env):

    def __init__(self):
        # Define the action space
        # The first action is the angle of rotation (-π to π)
        # The second action is the direction of movement (0: stop, 1: forward)
        self.action_space = spaces.Box(
                                        low=np.array([-np.pi, 0], dtype=np.float32),
                                        high=np.array([np.pi, 1], dtype=np.float32),
                                        dtype=np.float32
                                      )

        # Define the observation space
        # The observation space has 10 dimensions:
        # 1. Agent x-coordinate
        # 2. Agent y-coordinate
        # 3. Agent x-velocity
        # 4. Agent y-velocity
        # 5. Agent angle with respect to x-axis (-pi to pi)
        # 6. Ball x-coordinate
        # 7. Ball y-coordinate
        # 8. Ball x-velocity
        # 9. Ball y-velocity
        # 10. Goal x-coordinate
        # 11. Goal y-coordinate
        # 12. Goal x-coordinate(-)
        # 13. Goal y-coordinate(-)
        # 14. Obstacle x-coordinate
        # 15. Obstacle y-coordinate
        
        low = np.array([ -6.5, -4.625, -10, -10, -np.pi, -6.5, -4.625, -10, -10, 11.7, -1.25, -11.7, -1.25, 0, 0], dtype=np.float32)
        high = np.array([6.5, 4.625, 10, 10, np.pi, 6.5, 4.625, 10, 10, 11.7, 1.25,-11.7, 1.25,0, 0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
