from torch import nn

class RewardFromLatents(nn.Module):
    def __init__(self, latents_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latents_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)
