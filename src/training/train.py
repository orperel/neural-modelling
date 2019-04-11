import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from graphics.render.interactive import InteractiveWidget
from graphics.render.render_engine import RenderEngine
from graphics.render.renderable_mesh import RenderableMesh
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset
from app.events.visualize_pre_modifier import VisualizePreModifierEventHandler
from app.events.visualize_post_modifier import VisualizePostModifierEventHandler
from model_loader import load_model
from model.transformer.Optim import ScheduledOptim
from training.modifier_labels_decoder import ModifierLabelsDecoder


class Train:

    def __init__(self, config: dict, engine: RenderEngine):
        self.config = config
        self.debug_level = config['DEBUG_LEVEL']
        use_cuda = config['CUDA'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.engine = engine
        self.model = load_model(config)
        self.optimizer = self.initialize_optimizer(model=self.model)

        self.labels_decoder = ModifierLabelsDecoder()

    def train(self):

        epochs = self.config['TRAIN']['EPOCHS']

        train_dataloader = self.create_dataloader(config=self.config, engine=self.engine, data_type='train')
        test_dataloader = self.create_dataloader(config=self.config, engine=self.engine, data_type='test')

        for epoch_id in range(epochs):
            self.train_epoch(model=self.model,
                             train_dataloader=train_dataloader,
                             optimizer=self.optimizer)

    def load_dataset(self, dataset_config, engine):

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
        else:
            raise ValueError('Unknown dataset type encountered in config: %r' % (dataset_config['TYPE']))

        if 'show_gt_animations' in self.debug_level:  # Show animations
            dataset.on_pre_modifier_execution += VisualizePreModifierEventHandler(engine)
            dataset.on_post_modifier_execution += VisualizePostModifierEventHandler(engine)

        return dataset

    def create_dataloader(self, config, engine, data_type):
        data_type = data_type.upper()
        dataset_config = config[data_type]['DATASET']
        dataset = self.load_dataset(dataset_config, engine)
        shuffle = data_type == 'TRAIN'
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=config[data_type]['NUM_WORKERS'],
            batch_size=config[data_type]['BATCH_SIZE'],
            shuffle=shuffle)
        return dataloader

    def initialize_optimizer(self, model):

        optimizer_config = self.config['TRAIN']['OPTIMIZER']
        img2seq_config = self.config['MODEL']['IMG_TO_SEQ']

        optimizer_type = optimizer_config['TYPE']
        is_scheduled = optimizer_config['SCHEDULED']

        if optimizer_type == 'adam':
            betas = optimizer_config['BETAS']
            eps = float(optimizer_config['EPS'])

            optimizer = optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=betas, eps=eps
            )
        else:
            raise ValueError(f'Unsupported optimizer type {optimizer_type}')

        if is_scheduled:
            optimizer = ScheduledOptim(
                optimizer=optimizer,
                d_model=img2seq_config['D_MODEL'],
                n_warmup_steps=optimizer_config['N_WARMUP_STEPS']
            )

        return optimizer

    def calculate_loss(self, pred, modifiers):
        pred_modifier_class_id, \
            pred_element_type_tensor, \
            pred_element_pos_tensor, \
            pred_modifier_params = pred

        gt_modifier_class_id, \
            gt_element_type_tensor, \
            gt_element_pos_tensor, \
            element_pos_mask_tensor, \
            gt_modifier_params, \
            modifier_params_mask_tensor = self.labels_decoder.decode(modifiers)

        # Mask out irrelevant entries, as each modifier may need different params
        # Also make sure to pad the mask if needed, as the current batch may contain a group of modifiers whose
        # parameters don't amount to the maximal available (predictions always predict the maximum)
        elem_pos_dim_gap = pred_element_pos_tensor.shape[1] - element_pos_mask_tensor.shape[1]
        element_pos_mask_tensor = F.pad(element_pos_mask_tensor, (0, elem_pos_dim_gap))
        pred_element_pos_tensor.mul_(element_pos_mask_tensor)

        modifier_param_dim_gap = pred_modifier_params.shape[1] - modifier_params_mask_tensor.shape[1]
        modifier_params_mask_tensor = F.pad(modifier_params_mask_tensor, (0, modifier_param_dim_gap))
        pred_modifier_params.mul_(modifier_params_mask_tensor)

        class_id_loss = F.cross_entropy(pred_modifier_class_id, gt_modifier_class_id, reduction='sum')
        elem_type_loss = F.cross_entropy(pred_element_type_tensor, gt_element_type_tensor, reduction='sum')
        elem_pos_loss = F.mse_loss(pred_element_pos_tensor, gt_element_pos_tensor)
        mod_params_loss = F.mse_loss(pred_modifier_params, gt_modifier_params)

        loss = class_id_loss + elem_type_loss + elem_pos_loss + mod_params_loss

        return loss


    def train_epoch(self, model, train_dataloader, optimizer, smoothing=None):
        ''' Epoch operation in training phase'''

        device = self.device

        model.train()

        total_loss = 0
        total_batches = 0
        total_modifiers = 0

        for i, batch in enumerate(train_dataloader):

            # Input dimensions of Pytorch Tensors:
            # rendered_triplet - B x |I| x H x W x C where I is the number of reference images
            # modifiers - |M| x |max_length(m_i)| where m_i is some modifier in current set of modifiers: M
            batch = tuple(map(lambda x: x.to(device), batch))
            rendered_triplet, modifiers = batch

            # left_img, top_img, front_img = rendered_triplet
            optimizer.zero_grad()
            pred = self.model(rendered_triplet, modifiers)

            pred, modifiers = tuple(comp[1:, :] for comp in pred), modifiers[:, 1:, :]
            loss = self.calculate_loss(pred, modifiers)
            loss.backward()

            # update parameters
            optimizer.step_and_update_lr()

            batch_loss = loss.item()
            total_loss += batch_loss
            total_batches += batch[1].shape[0]
            total_modifiers += batch[1].shape[1]

            print(f'Batch {str(i)} ; Loss: {batch_loss:.2f} ; Total modifiers: {total_modifiers}')

        mean_epoch_loss = total_loss / total_batches
        mean_loss_per_modifier = total_loss / total_modifiers

        return total_loss, mean_epoch_loss, mean_loss_per_modifier
