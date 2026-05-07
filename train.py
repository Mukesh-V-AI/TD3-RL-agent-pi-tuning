import torch
import numpy as np
from env_pi import PIGainTuningEnv as PIEnv
from csac_agent import CSAC
from replay_buffer import ReplayBuffer

# ================= INIT =================
env = PIEnv()
agent = CSAC(6, 2)
buffer = ReplayBuffer()

EPISODES = 300
BATCH_SIZE = 64
WARMUP = 1000

# Disturbances for validation
TEST_DISTURBANCES = [0.05, 0.1, 0.15, 0.2, 0.3]

# ================= TRAIN LOOP =================
for ep in range(EPISODES):

    obs, _ = env.reset(); s = obs
    done = False

    ep_reward = 0
    steps = 0

    # ----------- TRAINING EPISODE -----------
    while not done:

        # Exploration
        if len(buffer.s) < WARMUP:
            action = np.random.uniform(-1, 1, size=2)
        else:
            action = agent.act(s)

        s2, r, term, trunc, info = env.step(action)
        done = term or trunc

        buffer.add(s, action, r, s2, done)

        # Update agent
        if len(buffer.s) > WARMUP:
            agent.update(buffer)

        s = s2
        ep_reward += r
        steps += 1

    # ================= TRAIN EPISODE SUMMARY =================
    OS = info["os"] * 100
    US = info["us"] * 100
    TS = info["ts"]
    SSE = abs(env.actual_output - env.setpoint)

    print(f"\nEPISODE {ep}")
    print(f"Reward       : {ep_reward:.2f}")
    print(f"Overshoot    : {OS:.2f} %")
    print(f"Undershoot   : {US:.2f} %")
    print(f"Settling Time: {TS:.2f} s")
    print(f"Final Error  : {SSE:.2f}")
    print(f"Steps        : {steps}")

    # ================= MULTI-DISTURBANCE VALIDATION =================
    print("\n--- VALIDATION ACROSS DISTURBANCES ---")

    all_ok = True

    for d in TEST_DISTURBANCES:

        obs, _ = env.reset(options={'disturbance': d}); s = obs
        done = False

        while not done:
            action = agent.act(s)
            s2, r, term, trunc, info = env.step(action)
            done = term or trunc
            s = s2

        OS = info["os"] * 100
        US = info["us"] * 100
        TS = info["ts"]
        SSE = abs(env.actual_output - env.setpoint)

        print(f"Dist {int(d*100)}% → OS:{OS:.2f} | US:{US:.2f} | TS:{TS:.2f} | SSE:{SSE:.2f}")

        # ----------- HARD CONSTRAINT CHECK -----------
        if OS > 15 or US > 15 or TS > 3 or SSE > 5:
            all_ok = False

    # ================= SAVE MODEL =================
    if all_ok:
        print("\n🔥 ALL CONSTRAINTS SATISFIED — SAVING MODEL 🔥")
        torch.save(agent.actor.state_dict(), "best_model.pth")
    else:
        print("\n❌ Constraints NOT satisfied yet")

    print("========================================")
