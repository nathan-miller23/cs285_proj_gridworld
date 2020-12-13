from agents import Agent
from utils import softmax
from deep_rl import information_radius_batched
import torch
import numpy as np

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

class AgentFromTorchEnsemble(Agent):

    def __init__(self, models, use_cuda=False):
        self.models = models
        self.use_cuda = use_cuda

    def _ensemble_probs(self, observation):
        # Return shape (len(self.models), self.action_space.n)
        preds = []
        for model in self.models:
            with torch.no_grad():
                obs = torch.tensor(observation).unsqueeze(0).permute(0, 3, 1, 2).float()
                if (self.use_cuda):
                    obs = obs.cuda()
                    logits = model(obs).cpu().numpy()
                else:
                    logits = model(obs).numpy()
            pred = softmax(logits).squeeze()
            preds.append(pred)
        return np.vstack(preds)

    def action_probs(self, observation):
        return self._ensemble_probs(observation).mean(0)

    def uncertainty(self, observation):
        ensemble_preds = self._ensemble_probs(observation)
        ensemble_preds = np.expand_dims(ensemble_preds, 1)
        info_rad = information_radius_batched(ensemble_preds, from_logits=False)
        return info_rad
