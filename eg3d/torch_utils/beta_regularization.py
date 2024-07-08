import torch
import torch.nn as nn
from torch.distributions import Beta

class BetaRegularizationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(BetaRegularizationLoss, self).__init__()
        self.beta_distribution = Beta(alpha, beta)
    
    def forward(self, opacities):
        # Ensure opacities are within (0, 1)
        opacities = torch.clamp(opacities, 1e-10, 1e-2)
        # Compute the negative log-likelihood
        nll = -self.beta_distribution.log_prob(opacities)
        # Return the mean loss
        return torch.mean(nll)

# # Example usage
# opacities = torch.tensor([0.1, 0.9, 0.5, 0.8], requires_grad=True)
# criterion = BetaRegularizationLoss()
# loss = criterion(opacities)
# loss.backward()
# print(loss.item())