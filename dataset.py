import torch
from torch.utils.data import Dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

import numpy as np

import random

from typing import Optional, List, Tuple

from processing.processor import BATTSProcessor
from processing.target import TargetBATTSProcessor

class BATTSDataset(Dataset):
    def __init__(self, manifest: str, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.table = pq.read_table(manifest, columns=['speaker', 'tokens', 'signal'])
        if num_examples is not None:
            self.table = self.table.slice(0, num_examples)
        index_col = pa.array(pa.range(self.table.num_rows))
        self.table.add_column(0, 'index', index_col)

    def __len__(self) -> int:
        return self.table.num_rows
    
    def get_row(self, index: int):
        return {col: self.table[col][index].as_py() for col in self.table.column_names}
    
    def get_ref_audio(self, speaker: str, index: int) -> List[float]:
        speaker_condition = pc.equal(self.table['speaker'], pc.scalar(speaker))
        index_condition = pc.not_equal(self.table['index'], pc.scalar(index))

        conditions = pc.and_(speaker_condition, index_condition)

        filtered_table = self.table.filter(conditions)
        randomed_index = random.randint(0, filtered_table.num_rows)

        return self.get_row(randomed_index)['signal']
    
    def __getitem__(self, index: int):
        row = self.get_row(index)

        tokens = torch.tensor(np.array(row['tokens']), dtype=torch.long)
        ref_signal = torch.tensor(np.array(self.get_ref_audio(row['speaker'], index)), dtype=torch.float)
        label = torch.tensor(np.array(row['signal']), dtype=torch.float)

        return tokens, ref_signal, label
    
class BATTSCollate:
    def __init__(self, input_processor: BATTSProcessor, target_processor: TargetBATTSProcessor) -> None:
        self.input_processor = input_processor
        self.target_processor = target_processor

    def __call__(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]):
        tokens, ref_signals, labels = zip(*batch)

        tokens, mels, se_signals, token_lengths = self.input_processor(tokens, ref_signals)
        labels, label_lengths = self.target_processor(labels)

        return tokens, mels, se_signals, labels, token_lengths, label_lengths