import torch
from networks import Actor
import numpy as np

# Load model once (IMPORTANT)
model = Actor(5,2)
model.load_state_dict(torch.load("actor.pth", map_location=torch.device('cpu')))
model.eval()

def get_action(state):
    state = np.array(state, dtype=np.float32)

    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0)
        action = model(s).squeeze(0).numpy()

    return action.tolist()