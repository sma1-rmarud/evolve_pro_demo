import torch
import torch.nn as nn
import torch.nn.functional as F

class ESM3WithFitnessHead(nn.Module):
    def __init__(self, esm3, d_model):
        super().__init__()
        self.esm3 = esm3
        # self.fitness_head = nn.Linear(d_model *3 ,1)
        hidden_dim = d_model
        self.proj = nn.Linear(d_model * 3, hidden_dim * 2)  # 3개 concat 하니까 크기 * 3, SwiGLU용
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        ss8_tokens=None,
        sasa_tokens=None,
        mut_sequence_tokens=None,
        mut_structure_coords=None,
        wt_sequence_tokens=None,
        wt_structure_coords=None,
        **kwargs,
    ):
        mut_out = self.esm3(
            sequence_tokens=mut_sequence_tokens,
            structure_coords=mut_structure_coords,
            ss8_tokens=ss8_tokens,
            sasa_tokens=sasa_tokens,
            **kwargs
        )
        mut_emb = mut_out.embeddings.mean(dim=1)  # (B, d_model)
        
        wt_out = self.esm3(
            sequence_tokens=wt_sequence_tokens,
            structure_coords=wt_structure_coords,
            ss8_tokens=ss8_tokens,
            sasa_tokens=sasa_tokens,
            **kwargs
        )

        wt_emb = wt_out.embeddings.mean(dim=1)  # (B, d_model)
        
        diff  = wt_emb - mut_emb
        
        combined = torch.cat((mut_emb, wt_emb, diff), dim=1)
        
        x1, x2 = self.proj(combined).chunk(2, dim=-1)  # (B, hidden), (B, hidden)
        x = F.silu(x1) * x2  # SwiGLU
        x = self.norm(x)
        return self.out(x).squeeze(-1)  # (B,)
