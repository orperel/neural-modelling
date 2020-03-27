import yaml
import os
import torch
from src.graphics.render.render_engine import RenderEngine
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset


def load_config():
    with open('configs/config.yaml', 'r') as yaml_file:
        config_file = yaml.load(yaml_file)
    return config_file


def store_yaml(filename, content):
    with open(filename, 'w') as yaml_file:
        yaml.dump(content, yaml_file, default_flow_style=False)


def create_dataset(engine, dataset_config):
    if dataset_config['TYPE'] == 'noisy_primitives':
        dataset = NoisyPrimitivesDataset(render_engine=engine,
                                         size=dataset_config['SIZE'],
                                         cache=False,
                                         min_modifier_steps=dataset_config['MIN_MODIFIER_STEPS'],
                                         max_modifier_steps=dataset_config['MAX_MODIFIER_STEPS'],
                                         modifiers_pool=dataset_config['MODIFIERS'],
                                         min_pertubration=dataset_config['MIN_PERTUBRATION'],
                                         max_pertubration=dataset_config['MAX_PERTUBRATION']
                                         )
    return dataset


def create_dataset_dir(datasets_root_path, dataset_name):

    if not os.path.exists(datasets_root_path):
        os.makedirs(datasets_root_path)

    available_datasets = os.listdir(datasets_root_path)

    running_idx = 1
    next_candidate_name = dataset_name + '_' + str(running_idx)

    while next_candidate_name in available_datasets:
        running_idx += 1
        next_candidate_name = dataset_name + '_' + str(running_idx)

    full_path = os.path.join(datasets_root_path, next_candidate_name)
    os.makedirs(full_path)
    return full_path


config = load_config()  # Use training configuration to get the requested dataset details
engine = RenderEngine()
dataset = create_dataset(engine=engine, dataset_config=config['TRAIN']['DATASET'])

dataset_name = str(dataset)
DATASETS_ROOT_DIR = os.path.join('/Users/orperel/personal/tau/neural-modelling/', 'data', 'generated')
dataset_dir = create_dataset_dir(DATASETS_ROOT_DIR, dataset_name)

meta_file_content = dataset.summary()
store_yaml(filename=os.path.join(dataset_dir, 'dataset_info.yaml'), content=meta_file_content)

data_entry_path_prefix = os.path.join(dataset_dir, 'data_')
labels_entry_path_prefix = os.path.join(dataset_dir, 'labels_')

for i in range(len(dataset)):
    rendered_triplet, modifiers = dataset.__getitem__(i)

    data_entry_path = data_entry_path_prefix + str(i) + '.pt'
    torch.save(rendered_triplet, data_entry_path)

    labels_entry_path = labels_entry_path_prefix + str(i) + '.pt'
    torch.save(modifiers, labels_entry_path)
