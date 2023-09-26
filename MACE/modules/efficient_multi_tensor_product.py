import torch
from e3nn import o3
from .efficient_utils import FFT_batch_channel, sh2fs_batch_channel, fs2sh_batch_channel

Y_dict = torch.load("/home/tlchen/mace-probe-eff/MACE/MACE/modules/coefficient_sh2fs.pt")
H_dict = torch.load("/home/tlchen/mace-probe-eff/MACE/MACE/modules/coefficient_fs2sh.pt")

class EfficientMultiTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.dimensions = irreps_in.count((0, 1))
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device

        Li = self.irreps_in.lmax + 1
        Lo = self.irreps_out.lmax + 1
        self.Li = Li
        self.Lo = Lo

        self.Y = Y_dict[Li].to(device)
        # self.Hs = {}
        lmaxs = torch.arange(2, correlation + 1) * (Li - 1)
        self.Hs = list(map(lambda lmax: H_dict[lmax + 1].to(device), lmaxs.tolist()))
        self.offsets_st = lmaxs - Lo + 1
        self.offsets_ed = lmaxs + Lo

        def gen_mask(L):
            left_indices = torch.arange(L, device=device).view(1, -1)  
            right_indices = torch.arange(L - 2, -1, -1, device=device).view(1, -1)  
            column_indices = torch.cat((left_indices, right_indices), dim=1).repeat(L, 1)  
            row_indices = torch.arange(L, device=device).view(-1, 1).repeat(1, 2 * L - 1)  
            mask = torch.abs(column_indices - (L - 1)) <= row_indices  
            mask2D = (torch.ones(L, 2 * L - 1, device=device) * mask).to(bool)
            return mask2D.flatten()
        self.mask_i, self.mask_o = list(map(gen_mask, [Li, Lo]))

        slices = 2 * torch.arange(Lo, device=device) + 1
        self.slices = slices.tolist()
        
        self.weights = torch.nn.ParameterDict({})
        for i in range(1, correlation + 1):
            w = torch.nn.Parameter(
                torch.randn(1, num_elements, self.dimensions, self.Lo)
            )
            self.weights[str(i)] = w
    
    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):

        # 3D to 4D
        n_nodes = atom_feat.shape[0]
        feat3D = torch.zeros(n_nodes, self.dimensions, self.mask_i.shape[0], device=self.device)
        feat3D[:, :, self.mask_i] = atom_feat
        feat4D = feat3D.reshape(n_nodes, self.dimensions, self.Li, -1) # (B, C, Li, 2Li-1)
        
        # efficient
        
        weights = (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1) # (B, C, Lo, 1)
        result = feat4D[:, :, :self.Lo, self.Li-self.Lo:self.Li+self.Lo-1] * weights
        
        fs_out = {}
        fs_out[1] = sh2fs_batch_channel(feat4D, self.Y)
        for nu in range(2, self.correlation + 1):
            if nu % 2 == 0:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2])
            else:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2 + 1])
            idx = nu - 2
            weights = (self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1)
            result += weights * fs2sh_batch_channel(fs_out[nu], self.Hs[idx]).real[:, :, :self.Lo, self.offsets_st[idx]:self.offsets_ed[idx]]
        
        # 4D to 2D
        result3D = result.reshape(n_nodes, self.dimensions, -1)[ :, :, self.mask_o]
        irreps = torch.split(result3D, self.slices, dim=-1)
        irreps_flatten = list(map(lambda x: x.flatten(start_dim=1), irreps))
        result2D = torch.cat(irreps_flatten, dim=-1)

        return result2D