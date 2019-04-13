import torch.nn.functional as F
from torch._six import container_abcs
from torch.utils.data.dataloader import default_collate


def modifiers_collate(batch):
    if isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [modifiers_collate(samples) for samples in transposed]

    max_modifiers_seq = max([entry.shape[0] for entry in batch])
    padded_batch = [F.pad(input=entry, pad=(0, 0, 0, max_modifiers_seq - entry.shape[0])) for entry in batch]
    return default_collate(padded_batch)
