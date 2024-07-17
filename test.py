#%%
from processing.processor import BATTSProcessor
# %%
processor = BATTSProcessor('./configs/vi.json')
# %%
processor.sentence2phonemes("trí nè.")
# %%
