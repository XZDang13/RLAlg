import torch

class Normalizer:
    def __init__(self, size, eps=1e-8, default_clip_range=float('inf'), device="cpu"):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = torch.zeros(self.size).to(device)
        self.local_sumsq = torch.zeros(self.size).to(device)
        self.local_count = torch.zeros(1).to(device)
        # get the total sum sumsq and sum count
        self.total_sum = torch.zeros(self.size).to(device)
        self.total_sumsq = torch.zeros(self.size).to(device)
        self.total_count = torch.ones(1).to(device)
        # get the mean and std
        self.mean = torch.zeros(self.size).to(device)
        self.std = torch.ones(self.size).to(device)

    def normalize(self, v, clip_range=None, update=False):
        if update:
            v = v.view(-1, self.size)
            # do the computing
            self.local_sum += v.sum(dim=0)
            self.local_sumsq += (v ** 2).sum(dim=0)
            self.local_count[0] += v.size(0)

        if clip_range is None:
            clip_range = self.default_clip_range
        return torch.clamp((v - self.mean) / (self.std), -clip_range, clip_range)
    
    def reset(self):
        self.local_count.fill_(0)
        self.local_sum.fill_(0)
        self.local_sumsq.fill_(0)

    def recompute_stats(self):
        local_count = self.local_count.clone()
        local_sum = self.local_sum.clone()
        local_sumsq = self.local_sumsq.clone()
        
        self.local_count.fill_(0)
        self.local_sum.fill_(0)
        self.local_sumsq.fill_(0)
        # sync the stats
        # update the total stuff
        self.total_sum += local_sum
        self.total_sumsq += local_sumsq
        self.total_count += local_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        eps_tensor = torch.tensor(self.eps**2)
        self.std = torch.sqrt(torch.maximum(eps_tensor, (self.total_sumsq / self.total_count) - (self.total_sum / self.total_count)**2))

    def state_dict(self):
        state_dict = {
            "local_sum": self.local_sum,
            "local_sumsq": self.local_sumsq,
            "local_count": self.local_count,
            "total_sum": self.total_sum,
            "total_sumsq": self.total_sumsq,
            "total_count": self.total_count,
            "mean": self.mean,
            "std": self.std
        }

        return state_dict
    
    def load_state_dict(self, state_dict):
        self.local_sum = state_dict["local_sum"]
        self.local_sumsq = state_dict["local_sumsq"]
        self.local_count = state_dict["local_count"]
        self.total_sum = state_dict["total_sum"]
        self.total_sumsq = state_dict["total_sumsq"]
        self.total_count = state_dict["total_count"]
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]