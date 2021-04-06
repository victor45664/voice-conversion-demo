

import torch

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class StatisticPool(torch.nn.Module):
    def __init__(self,EPSILON=1e-5):
        super(StatisticPool, self).__init__()
        self.EPSILON=EPSILON

    def forward(self,x,seq_length):    #x[B,T,D]
        D=x.size(2)
        mask=get_mask_from_lengths(seq_length)
        mask = mask.unsqueeze(2) # [N, T,1]

        mask = mask.expand(mask.size(0), mask.size(1),D)
        x=x*mask

        x_sum=torch.sum(x,dim=1).transpose(0,1)
        x_mean=(x_sum/seq_length).transpose(0,1)

        x_cut_mean=x-x_mean.unsqueeze(1)
        x_cut_mean=x_cut_mean*mask

        x_var=torch.square(x_cut_mean).sum(1).transpose(0,1)
        x_std = torch.sqrt((x_var / seq_length).transpose(0, 1)+self.EPSILON)

        return x_mean,x_std  # [B,D]

class StatisticsReplacementLayer(torch.nn.Module):
    def __init__(self):
        super(StatisticsReplacementLayer, self).__init__()
        self.sp=StatisticPool()

    def forward(self,x,seq_length,target_mean, target_std):



        s_mean,s_std=self.sp(x,seq_length)
        norm_x=(x-s_mean.unsqueeze(1))/s_std.unsqueeze(1)
        conditioned_x=norm_x*target_std.unsqueeze(1)+target_mean.unsqueeze(1)



        return conditioned_x




if __name__ == '__main__':
    import numpy as np
    SR=StatisticsReplacementLayer()
    SP=StatisticPool()
    B=2
    max_length=55
    dim=5

    hidden_representations=torch.rand((B, max_length, dim))

    seq_length=torch.randint(1, max_length, (B,))
    seq_length[0 ]=max_length

    mean, std = SP(hidden_representations, seq_length)
    print(mean,std)   # original mean and std


    target_mean=torch.from_numpy(np.zeros((2, dim))+5)
    target_std=torch.from_numpy(np.ones((2, dim))*1.5)

    conditioned_x = SR(hidden_representations, seq_length, target_mean, target_std)


    mean,std=SP(conditioned_x, seq_length)


    print(mean,std)   # to Verify the mean and std is correctly changed






