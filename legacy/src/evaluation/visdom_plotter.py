import sys
import os
import numpy as np
from visdom import Visdom


class VisdomPlotter:
    """
    A Visdom based plotter, to plot aggregated metrics.

    How to use:
    ------------
    (1) Start the server with:
            python -m visdom.server
    (2) Then, in your browser, you can go to:
            http://localhost:8097
    """

    def __init__(self, experiment_env, server='http://localhost', port=8097):
        self.server = server
        self.port = port
        self.viz = Visdom(server=server, port=port)  # Connect to Visdom server on server / port
        if not self.start_visdom_server():
            raise ValueError('Failed to launch Visdom server at %r:%r' % (server, port))

        if experiment_env in self.viz.get_env_list():
            self.viz.delete_env(experiment_env)  # Clear previous runs with same id
        self.experiment_env = experiment_env
        self.plots = {}

    def start_visdom_server(self):
        is_visdom_server_connected = self.viz.check_connection(timeout_seconds=1)   # Ping if it's already on..
        if not is_visdom_server_connected:
            interpreter_path = sys.executable
            os.system(interpreter_path + ' -m visdom.server &')
            is_visdom_server_connected = self.viz.check_connection(timeout_seconds=35)
        return is_visdom_server_connected

    def plot_single_metric(self, metric, line_id, title, epoch, value):

        if metric not in self.plots:
            self.plots[metric] = self.viz.line(X=np.array([epoch, epoch]), Y=np.array([value, value]),
                                               env=self.experiment_env,
                                               opts=dict(
                                                legend=[line_id],
                                                title=title,
                                                xlabel='Epochs',
                                                ylabel=metric
            ))
        else:
            self.viz.line(X=np.array([epoch]),
                          Y=np.array([value]),
                          env=self.experiment_env,
                          win=self.plots[metric],
                          name=line_id,
                          update='append')

    def plot_confusion_matrix(self, metric, matrix, label_classes):
        if metric not in self.plots:
            self.plots[metric] = self.viz.heatmap(X=matrix, env=self.experiment_env,
                                                  opts=dict(
                                                      columnnames=label_classes,
                                                      rownames=label_classes
                                                  ))
        else:
            self.viz.heatmap(X=matrix, env=self.experiment_env, win=self.plots[metric],
                             opts=dict(
                                 columnnames=label_classes,
                                 rownames=label_classes
                             ))

    def plot_images(self, images_bchw):
        self.viz.images(images_bchw)

    def plot_aggregated_metrics(self, metrics, epoch):

        for metric in metrics.metrics:
            title = metrics.metric_to_title[metric]
            value = metrics[epoch][metric]

            if metric == 'confusion_matrix':
                label_classes = metrics.label_classes
                self.plot_confusion_matrix(metric, value, label_classes)
            else:
                if hasattr(value, 'shape') and value.size > 1:
                    for idx, dim_val in enumerate(value):
                        line_id = metrics.label_classes[idx]
                        self.plot_single_metric(metric, line_id, title, epoch, dim_val)
                else:
                    line_id = metrics.data_type
                    self.plot_single_metric(metric, line_id, title, epoch, value)
