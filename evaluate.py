from env_pi import PIEnv

env = PIEnv()

for d in [0.05,0.1,0.15,0.2,0.3]:

    env.dist = d
    s = env.reset()

    done = False

    while not done:
        a = [0,0]  # replace with trained agent
        s, r, done, info = env.step(a)

    print("Dist:", d, info)