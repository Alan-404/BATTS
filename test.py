#%%
import torch
from model.modules.gpt import GPT
# %%
model = GPT(n_tokens=1000, n_blocks=12, model_dim=512, n_heads=8)
# %%
model
# %%
x = torch.tensor([[1,2,3,4,5], [1,7,3,4,6]])
hidden = torch.rand((2, 10, 512))
# %%
out = model(x, hidden)
# %%
out.shape
# %%
