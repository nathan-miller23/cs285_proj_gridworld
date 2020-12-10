from agents import Agent
from utils import softmax
import torch

class AgentFromTorch(Agent):

    def __init__(self, model):
        self.model = model

    def action_probs(self, observation):
        with torch.no_grad():
            obs = torch.tensor(observation).unsqueeze(0).permute(0, 3, 1, 2).float()
            logits = self.model(obs).numpy()
        return softmax(logits).squeeze()
