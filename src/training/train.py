import torch
import torch.optim as optim
import torch.nn.functional as F
import traceback
from graphics.render.render_engine import RenderEngine
from src.data.data_utils import modifiers_collate
from src.data.noisy_primitives_dataset import NoisyPrimitivesDataset
from src.data.pregenerated_dataset import PregeneratedDataset
from app.events.visualize_pre_modifier import VisualizePreModifierEventHandler
from app.events.visualize_post_modifier import VisualizePostModifierEventHandler
from app.modifier_visitors.modifier_id_visitor import ModifierEnum
from app.modifier_visitors.selected_element_type_visitor import ElementType
from model_loader import load_model
from model.transformer.Optim import ScheduledOptim
from training.modifier_labels_decoder import ModifierLabelsDecoder
from evaluation.metrics import ExperimentMetrics
from evaluation.plotter import Plotter


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
        self.train_metrics = ExperimentMetrics(modifier_class_labels=[m.name for m in ModifierEnum],
                                               element_type_labels=[e.name for e in ElementType],
                                               data_type='Training')
        self.plotter = Plotter(experiment_env=config['EXPERIMENT_NAME'])

    def train(self):

        epochs = self.config['TRAIN']['EPOCHS']

        train_dataloader = self.create_dataloader(config=self.config, engine=self.engine, data_type='train')
        test_dataloader = self.create_dataloader(config=self.config, engine=self.engine, data_type='test')

        best_train_loss = float("inf")

        for epoch_id in range(1, epochs+1):
            mean_epoch_loss = self.train_epoch(model=self.model, train_dataloader=train_dataloader,
                                               optimizer=self.optimizer, epoch_id=epoch_id)

            if mean_epoch_loss < best_train_loss:
                best_train_loss = mean_epoch_loss
                torch.save(self.model, 'best_train_model.pt')

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
        elif dataset_config['TYPE'] == 'pregenerated':
            dataset = PregeneratedDataset(dataset_path=dataset_config['PATH'],
                                          modifiers_dim=dataset_config['MODIFIERS_DIM'])
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
            shuffle=shuffle,
            collate_fn=modifiers_collate)
        return dataloader

    def initialize_optimizer(self, model):

        optimizer_config = self.config['TRAIN']['OPTIMIZER']
        img2seq_config = self.config['MODEL']['IMG_TO_SEQ']

        optimizer_type = optimizer_config['TYPE']
        is_scheduled = optimizer_config['SCHEDULED']

        if optimizer_type == 'adam':
            lr = float(optimizer_config['LR'])
            betas = optimizer_config['BETAS']
            eps = float(optimizer_config['EPS'])

            optimizer = optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=lr, betas=betas, eps=eps
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

    def log_loss(self, class_id_loss, elem_type_loss, elem_pos_loss, mod_params_loss,
                 pred_modifier_class_id, pred_element_type_tensor, pred_element_pos_tensor, pred_modifier_params,
                 gt_modifier_class_id, gt_element_type_tensor, gt_element_pos_tensor, gt_modifier_params):

        print('-------------------------------------------------------------------')
        print(f'class_id_loss {class_id_loss.item()} ; ' +
              f'elem_type_loss {elem_type_loss.item()} ; ' +
              f'elem_pos_loss {elem_pos_loss.item()} ; ' +
              f'mod_params_loss {mod_params_loss.item()}')

        print(f'gt_modifier_class: {gt_modifier_class_id} | pred_modifier_class: {pred_modifier_class_id}')
        print(f'gt_element_type_tensor: {gt_element_type_tensor} | pred_element_type_tensor: {pred_element_type_tensor}')
        print(f'gt_element_pos_tensor: {gt_element_pos_tensor} | pred_element_pos_tensor: {pred_element_pos_tensor}')
        print(f'gt_modifier_params: {gt_modifier_params} | pred_modifier_params: {pred_modifier_params}')
        print('-------------------------------------------------------------------')

    def _override_null_target_for_cross_entropy(self, target_tensor, non_padded_modifiers_mask):
        """
        For padded entries, target tensor containing ground truth must contain a spceific value for the cross_entropy
        loss function to ignore it.
        :param target_tensor: Ground truth target tensor
        :param non_padded_modifiers_mask: Binary rows mask - where zero bit rows should be ignored
        :return: Adjusted target tensor, with masked values replaced by the special ignored index
        """
        non_padded_modifiers_mask = non_padded_modifiers_mask.long()
        real_gt_values = (target_tensor * non_padded_modifiers_mask)
        ignored_indices = (non_padded_modifiers_mask == 0).nonzero().squeeze()
        ignored_mask = torch.zeros_like(real_gt_values).scatter(0, ignored_indices, 1)
        ignored_values = ignored_mask * self.labels_decoder.modifiers_ignored_index

        return (real_gt_values + ignored_values).to(dtype=target_tensor.dtype)

    def _override_null_target_for_mse(self, target_tensor, non_padded_modifiers_mask):
        non_padded_modifiers_mask = non_padded_modifiers_mask.float()
        non_padded_modifiers_mask = non_padded_modifiers_mask.view(-1, 1).repeat(1, target_tensor.shape[1])

        return (target_tensor * non_padded_modifiers_mask).to(dtype=target_tensor.dtype)

    def calculate_loss(self, pred, modifiers, non_padded_modifiers_mask):
        # First unpack all prediction vectors
        pred_modifier_class_id, \
            pred_element_type_tensor, \
            pred_element_pos_tensor, \
            pred_modifier_params = pred

        # And decode all label vectors, as well as relevant masks,
        # since for each modifier not all dimensions are used
        gt_modifier_class_id, \
            gt_element_type_tensor, \
            gt_element_pos_tensor, \
            element_pos_mask_tensor, \
            gt_modifier_params, \
            modifier_params_mask_tensor = self.labels_decoder.decode(modifiers)

        # Now we apply the modifier dimensions masks:
        # Mask out irrelevant entries, as each modifier may need different params.
        # (for example: translate vertex needs 3 coordinates, translate face needs 9 coordinates, etc).
        # Also make sure to pad the mask if needed, as the current batch may contain a group of modifiers whose
        # parameters don't amount to the maximal available (predictions always predict the maximum amount of dimensions)
        elem_pos_dim_gap = pred_element_pos_tensor.shape[1] - element_pos_mask_tensor.shape[1]
        element_pos_mask_tensor = F.pad(element_pos_mask_tensor, (0, elem_pos_dim_gap))
        gt_element_pos_tensor = F.pad(gt_element_pos_tensor, (0, elem_pos_dim_gap))
        pred_element_pos_tensor = pred_element_pos_tensor.mul(element_pos_mask_tensor)

        modifier_param_dim_gap = pred_modifier_params.shape[1] - modifier_params_mask_tensor.shape[1]
        modifier_params_mask_tensor = F.pad(modifier_params_mask_tensor, (0, modifier_param_dim_gap))
        gt_modifier_params = F.pad(gt_modifier_params, (0, modifier_param_dim_gap))
        pred_modifier_params = pred_modifier_params.mul(modifier_params_mask_tensor)

        # Next - make sure ground truth labels are updated to ignore null entries: BOS modifiers and padding modifiers.
        # Make sure ground truth - target labels are contain null values that will block loss calculation
        # for irrelevant entries.
        # For CrossEntropyLoss, we use a special ignore_index value.
        # For MSELoss, we make sure all relevant dimensions are zero.
        gt_modifier_class_id = self._override_null_target_for_cross_entropy(gt_modifier_class_id,
                                                                            non_padded_modifiers_mask)
        gt_element_type_tensor = self._override_null_target_for_cross_entropy(gt_element_type_tensor,
                                                                              non_padded_modifiers_mask)
        gt_element_pos_tensor = self._override_null_target_for_mse(gt_element_pos_tensor,
                                                                   non_padded_modifiers_mask)
        gt_modifier_params = self._override_null_target_for_mse(gt_modifier_params,
                                                                non_padded_modifiers_mask)

        class_id_loss = F.cross_entropy(pred_modifier_class_id, gt_modifier_class_id,
                                        reduction='sum', ignore_index=self.labels_decoder.modifiers_ignored_index)
        elem_type_loss = F.cross_entropy(pred_element_type_tensor, gt_element_type_tensor,
                                         reduction='sum', ignore_index=self.labels_decoder.modifiers_ignored_index)
        elem_pos_loss = F.mse_loss(pred_element_pos_tensor, gt_element_pos_tensor, reduction='sum')
        mod_params_loss = F.mse_loss(pred_modifier_params, gt_modifier_params, reduction='sum')

        loss = class_id_loss + elem_type_loss + elem_pos_loss + mod_params_loss

        # self.log_loss(class_id_loss, elem_type_loss, elem_pos_loss, mod_params_loss,
        #               pred_modifier_class_id, pred_element_type_tensor, pred_element_pos_tensor, pred_modifier_params,
        #               gt_modifier_class_id, gt_element_type_tensor, gt_element_pos_tensor, gt_modifier_params)

        log_losses = {
            'total_loss': loss,
            'modifier_class_loss': class_id_loss,
            'selected_element_type_loss': elem_type_loss,
            'selected_element_pos_loss': elem_pos_loss,
            'modifier_params_loss': mod_params_loss
        }

        return loss, log_losses

    def train_epoch(self, model, train_dataloader, optimizer, epoch_id):
        ''' Epoch operation in training phase'''

        device = self.device

        model.train()

        total_loss = 0
        total_batches = 0
        total_modifiers = 0

        for i, batch in enumerate(train_dataloader):

            try:
                # Input dimensions of Pytorch Tensors:
                # rendered_triplet - B x |I| x H x W x C where I is the number of reference images
                # modifiers - |M| x |max_length(m_i)| where m_i is some modifier in current set of modifiers: M
                batch = tuple(map(lambda x: x.to(device), batch))
                rendered_triplet, modifiers = batch

                non_padded_modifiers_mask = self.labels_decoder.mask_padded_modifiers(modifiers)
                optimizer.zero_grad()
                pred = self.model(rendered_triplet, modifiers, non_padded_modifiers_mask)

                loss, log_losses = self.calculate_loss(pred, modifiers, non_padded_modifiers_mask)
                loss.backward()

                # Update parameters: support the case of scheduled optimizer here
                # (needed by some modules, such as the Transformer)
                if hasattr(optimizer, 'step_and_update_lr'):
                    optimizer.step_and_update_lr()
                else:
                    optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                total_batches += batch[1].shape[0]
                total_modifiers += batch[1].shape[1]
                print(f'Epoch {epoch_id} ; Batch {str(i)} ; Loss: {batch_loss:.2f} ; Total modifiers: {total_modifiers}')

                self.train_metrics.report_batch_results(epoch=epoch_id,
                                                        preds=pred,
                                                        labels=modifiers,
                                                        losses={l_name: l.item() for l_name, l in log_losses.items()},
                                                        total_batches=total_batches,
                                                        total_modifiers=total_modifiers)

                if epoch_id % self.config['LOG_RATE'] == 0:
                    self.plotter.plot_aggregated_metrics(self.train_metrics, epoch_id)


            except Exception as e:
                print(f'Exception have occured: {e}')
                traceback.print_exc()
                print('Skipping entry..')

        mean_epoch_loss = total_loss / total_batches

        if epoch_id % self.config['SAVE_RATE'] == 0:
            torch.save(model, 'latest_model.pt')

        return mean_epoch_loss
