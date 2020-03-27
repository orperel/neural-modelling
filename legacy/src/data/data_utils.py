import torch
import torch.nn.functional as F
from torch._six import container_abcs
from torch.utils.data.dataloader import default_collate
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset
from src.data.pregenerated_dataset import PregeneratedDataset
from src.app import VisualizePreModifierEventHandler
from src.app.events.visualize_post_modifier import VisualizePostModifierEventHandler


def modifiers_collate(batch):
    if isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [modifiers_collate(samples) for samples in transposed]

    max_modifiers_seq = max([entry.shape[0] for entry in batch])
    padded_batch = [F.pad(input=entry, pad=(0, 0, 0, max_modifiers_seq - entry.shape[0])) for entry in batch]
    return default_collate(padded_batch)


def load_dataset(dataset_config, debug_level, engine=None):

    if dataset_config['TYPE'] == 'noisy_primitives':
        dataset = NoisyPrimitivesDataset(render_engine=engine,
                                         size=dataset_config['SIZE'],
                                         cache=dataset_config['CACHE'],
                                         min_modifier_steps=dataset_config['MIN_MODIFIER_STEPS'],
                                         max_modifier_steps=dataset_config['MAX_MODIFIER_STEPS'],
                                         modifiers_pool=dataset_config['MODIFIERS'],
                                         min_pertubration=dataset_config['MIN_PERTUBRATION'],
                                         max_pertubration=dataset_config['MAX_PERTUBRATION']
                                         )
    elif dataset_config['TYPE'] == 'pregenerated':
        dataset = PregeneratedDataset(dataset_path=dataset_config['PATH'],
                                      modifiers_dim=dataset_config['MODIFIERS_DIM'])
    else:
        raise ValueError('Unknown dataset type encountered in config: %r' % (dataset_config['TYPE']))

    if 'show_gt_animations' in debug_level:  # Show animations
        dataset.on_pre_modifier_execution += VisualizePreModifierEventHandler(engine)
        dataset.on_post_modifier_execution += VisualizePostModifierEventHandler(engine)

    return dataset


def create_dataloader(config, data_type, engine=None):
    debug_level = config['DEBUG_LEVEL']
    data_type = data_type.upper()
    dataset_config = config[data_type]['DATASET']
    dataset = load_dataset(dataset_config, debug_level, engine)
    shuffle = data_type == 'TRAIN'
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config[data_type]['NUM_WORKERS'],
        batch_size=config[data_type]['BATCH_SIZE'],
        shuffle=shuffle,
        collate_fn=modifiers_collate)
    return dataloader
