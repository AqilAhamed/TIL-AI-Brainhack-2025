class RLManager:

    def __init__(self):
        # You can initialize internal state here if needed
        self.last_action = 0

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take.
        """

        # Example logic:
        # If scout == 1, act differently than scout == 0
        # Direction: 0=up, 1=right, 2=down, 3=left
        direction = observation.get("direction", 0)
        step = observation.get("step", 0)
        scout = observation.get("scout", 0)

        # Simple strategy:
        # Scout alternates moving right (1) and down (2)
        # Others rotate actions: up (0), right (1), down (2), left (3)
        if scout:
            return 1 if step % 2 == 0 else 2
        else:
            return step % 4