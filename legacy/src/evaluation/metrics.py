import sys
import logging
import numpy as np
import torch
from src.evaluation.metrics_decorators import handles_prediction, metric


class ExperimentMetrics:
    """
    A class for aggregating metrics over training / evaluation process.
    """

    # All supported metrics & their titles
    metric_to_title = {
        'avg_loss_per_batch': 'Average entry loss (per epoch)',
        'avg_loss_per_modifier': 'Average entry loss (per modifiers count in epoch)',
        'max_batch_loss': 'Maximum batch loss (per epoch)',

        'modifier_class_loss': 'Average modifier class loss',
        'selected_element_type_loss': 'Average Selected element type loss',
        'selected_element_pos_loss': 'Average selected element pos. loss',
        'modifier_params_loss': 'Average modifiers parameters loss',

        # 'modifier_class_confusion_matrix': 'Modifier class confusion matrix (GT / PRED)',
        # 'element_type_confusion_matrix': 'Modifier class confusion matrix (GT / PRED)'
    }

    eps = sys.float_info.epsilon

    def __init__(self, modifier_class_labels, element_type_labels, metrics=None, data_type='Training'):
        """
        :param modifier_class_labels: List of label names (each index corresponds to the label id as the model knows it)
        :param element_type_labels: List of label names (each index corresponds to the label id as the model knows it)
        :param metrics: List of metrics to aggregate (keys of ExperimentMetrics.metric_to_title)
        :param data_type: 'Training' or 'Test'
        """
        self.logger = logging.getLogger('neural-modelling')
        self.statistics_per_epoch = []
        self.modifier_class_labels = modifier_class_labels
        self.element_type_labels = element_type_labels
        self.metrics = metrics or [metric for metric in ExperimentMetrics.metric_to_title.keys()]  # Default: all
        self.data_type = data_type

    def _fetch_epoch_statistics(self, epoch):
        """
        Get the current epoch aggregated statistics.
        If this is the first time the method is invoked for this epoch, the metrics entry will be initialized.
        :param epoch: Epoch number (one indexed)
        :return: Dictionary containing aggregated metrics for given epoch
        """
        if len(self.statistics_per_epoch) < epoch:
            num_modifier_classes = len(self.modifier_class_labels)
            num_element_classes = len(self.element_type_labels)
            epoch_statistics = {
                'total_batches': 0,
                'total_modifiers': 0,
                'modifier_classes_seen': [0 for _ in self.modifier_class_labels],
                'element_types_seen': [0 for _ in self.element_type_labels],
                'total_loss': 0.0,
                'total_modifier_class_loss': 0.0,
                'total_selected_element_type_loss': 0.0,
                'total_selected_element_pos_loss': 0.0,
                'total_modifier_params_loss': 0.0,

                'avg_loss_per_epoch': 0.0,
                'avg_loss_per_modifier': 0.0,
                'max_batch_loss': 0.0,
                'modifier_class_loss': 0.0,
                'selected_element_type_loss': 0.0,
                'selected_element_pos_loss': 0.0,
                'modifier_params_loss': 0.0,
                'modifier_class_confusion_matrix': torch.zeros(num_modifier_classes, num_modifier_classes),
                'element_type_confusion_matrix': torch.zeros(num_element_classes, num_element_classes)
            }
            self.statistics_per_epoch.append(epoch_statistics)
        return self.statistics_per_epoch[epoch-1]

    @metric(name='avg_loss_per_batch')
    def _report_avg_loss_per_batch(self, epoch_statistics):
        epoch_statistics['avg_loss_per_batch'] = epoch_statistics['total_loss'] / epoch_statistics['total_batches']

    @metric(name='avg_loss_per_modifier')
    def _report_avg_loss_per_modifier(self, epoch_statistics):
        epoch_statistics['avg_loss_per_modifier'] = epoch_statistics['total_loss'] / epoch_statistics['total_modifiers']

    @metric(name='modifier_class_loss')
    def _report_modifier_class_loss(self, epoch_statistics):
        epoch_statistics['modifier_class_loss'] = \
            epoch_statistics['total_modifier_class_loss'] / epoch_statistics['total_batches']

    @metric(name='selected_element_type_loss')
    def _report_selected_element_type_loss(self, epoch_statistics):
        epoch_statistics['selected_element_type_loss'] = \
            epoch_statistics['total_selected_element_type_loss'] / epoch_statistics['total_batches']

    @metric(name='selected_element_pos_loss')
    def _report_selected_element_pos_loss(self, epoch_statistics):
        epoch_statistics['selected_element_pos_loss'] = \
            epoch_statistics['total_selected_element_pos_loss'] / epoch_statistics['total_batches']

    @metric(name='modifier_params_loss')
    def _report_modifier_params_loss(self, epoch_statistics):
        epoch_statistics['modifier_params_loss'] = \
            epoch_statistics['total_modifier_params_loss'] / epoch_statistics['total_batches']

    @metric(name='max_batch_loss')
    def _report_max_loss(self, epoch_statistics, loss):
        epoch_statistics['max_batch_loss'] = max(epoch_statistics['max_batch_loss'], loss)

    def _report_total_processed(self, epoch_statistics, total_batches, total_modifiers, losses):
        epoch_statistics['total_batches'] += total_batches
        epoch_statistics['total_modifiers'] += total_modifiers

        epoch_statistics['total_loss'] += losses['total_loss']
        epoch_statistics['total_modifier_class_loss'] += losses['modifier_class_loss']
        epoch_statistics['total_selected_element_type_loss'] += losses['selected_element_type_loss']
        epoch_statistics['total_selected_element_pos_loss'] += losses['selected_element_pos_loss']
        epoch_statistics['total_modifier_params_loss'] += losses['modifier_params_loss']

    def _update_confusion_matrix(self, epoch_statistics, labels, preds):
        """
        Updates the confusion matrix of the current predictions vs labels.
        E.g: A matrix of num_classes x num_classes which describes the amount of times each prediction was mistaken
        for another label. The matrix appears like so:

                            Prediction
                          ----------------------
                          | 0 | 1 | .... | C-1 |
                    ----------------------------
       Ground Truth | 0   |   |   | .... |     |
                    ----------------------------
                    | 1   |   |   | .... |     |
                    ----------------------------
                    | ... |  ...............   |
                    ----------------------------
                    | C-1 |   |   | .... |     |
                    ----------------------------

        See: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

        :param labels: Ground truth labels for the current batch
        :param preds: Predictions of the model for the current batch
        :return: Confusion matrix as PyTorch Tensor: (num_classes x num_classes)
        """
        with torch.no_grad():
            predicted_labels = torch.argmax(input=preds.data, dim=1)
            confusion_matrix = epoch_statistics['confusion_matrix']
            for t, p in zip(labels.view(-1), predicted_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    @handles_prediction
    def report_batch_results(self, epoch, preds, labels, losses, total_batches, total_modifiers):
        epoch_statistics = self._fetch_epoch_statistics(epoch)

        self._report_total_processed(epoch_statistics, total_batches, total_modifiers, losses)
        self._report_avg_loss_per_batch(epoch_statistics)
        self._report_avg_loss_per_modifier(epoch_statistics)
        self._report_modifier_class_loss(epoch_statistics)
        self._report_selected_element_type_loss(epoch_statistics)
        self._report_selected_element_pos_loss(epoch_statistics)
        self._report_modifier_params_loss(epoch_statistics)
        self._report_max_loss(epoch_statistics, loss=losses['total_loss'])

    def log_metrics(self, epoch):
        np.set_printoptions(precision=3)
        epoch_statistics = self._fetch_epoch_statistics(epoch)
        self.logger.info('Metrics for %r - epoch #%r:' % (self.data_type, epoch))
        self.logger.info('-------------------------')
        self.logger.info('- Entries processed: %r' % epoch_statistics['entries_seen'])

        for metric in self.metrics:
            metric_title = self.metric_to_title[metric]
            metric_value = epoch_statistics[metric]
            if hasattr(metric_value, 'size'):
                if isinstance(metric_value.size, int):
                    if metric_value.size > 1:
                        self.logger.info('- %r: %r' % (metric_title, str(metric_value)))
                    else:
                        self.logger.info('- %r: %.3f' % (metric_title, metric_value))
                elif len(metric_value.size()) == 2:
                    self.logger.info('- %r:\n %r' % (metric_title, metric_value))
            else:
                self.logger.info('- %r: %.3f' % (metric_title, metric_value))

    def __getitem__(self, epoch):
        return self.statistics_per_epoch[epoch - 1]

    def __iter__(self):
        return iter(self.statistics_per_epoch)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.logger = logging.getLogger('neural-modelling')
