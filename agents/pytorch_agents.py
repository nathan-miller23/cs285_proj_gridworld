from agents import Agent
from utils import softmax
import torch

class AgentFromTorch(Agent):

    def __init__(self, model, use_cuda=False):
        self.model = model
        self.use_cuda = use_cuda

    def action_probs(self, observation):
        with torch.no_grad():
            obs = torch.tensor(observation).unsqueeze(0).permute(0, 3, 1, 2).float()
            if (self.use_cuda):
                obs = obs.cuda()
                logits = self.model(obs).cpu().numpy()
            else:
                logits = self.model(obs).numpy()
        return softmax(logits).squeeze()
