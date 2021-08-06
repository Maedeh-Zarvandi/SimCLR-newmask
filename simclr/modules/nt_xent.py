import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

###########################################
# Here it comes the new mask function
    def mask_most_sim(self, sim):
        N = 2 * self.batch_size * self.world_size
        mask = torch.clone(input = self.mask)

        most_sim = torch.argsort(sim, dim=1, descending=True)
        
        pairs = {}
        for i, row in enumerate(most_sim):
            if i in pairs or (i + self.batch_size) % N in pairs:
                continue
            found = False
            for j in row:
                j = j.detach().cpu().item()
                if found:
                    break
                if j == i or j == (i + self.batch_size) % N:
                    continue
                if j in pairs or (j + self.batch_size) % N in pairs:
                    continue
                pairs[i] = pairs[(i + self.batch_size) % N] = [j, (j + self.batch_size) % N]
                pairs[j] = pairs[(j + self.batch_size) % N] = [i, (i + self.batch_size) % N]
                found = True

        for i in pairs:
            a, b = pairs[i]
            mask[i, a] = mask[i, b] = 0
            #mask[a, i] = mask[b, i] = 0

        return mask
###########################################

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature #torch.Size([256, 256])
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_most_sim(sim) #torch.Size([256, 256])
        negative_samples = sim[mask].reshape(N, -1)
        # negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
