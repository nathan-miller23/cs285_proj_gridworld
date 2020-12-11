from gym_minigrid.envs.tightrope import TightRope
from gym_minigrid.minigrid import Direction

env = TightRope()

while True:
    _ = env.reset()
    done = False
    while not done:
        env.render()
        key_input = input("action: ")
        if key_input == "s":
            action = Direction.down
        elif key_input == "a":
            action = Direction.left
        elif key_input == "w":
            action = Direction.up
        elif key_input == "d":
            action = Direction.right
        else:
            action = Direction.stay

        _, _, done, _ = env.step(action)

