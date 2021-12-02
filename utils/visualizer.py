import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
import visdom


class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.

        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
        self.vis = visdom.Visdom('http://76.71.152.113', port=configuration["port"])
        if not self.vis.check_connection():
            self.create_visdom_connections()


    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_current_semi_data_used(self, epoch, semi_data_used):
        """Display the current semi-supervised correct percent of labels on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            semi_data_used: The percentage of semi_data used and stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'sdu_plot_data'):
            self.sdu_plot_data = {'X': [], 'Y': [], 'legend': list(semi_data_used.keys())}
        self.sdu_plot_data['X'].append(epoch)
        self.sdu_plot_data['Y'].append([semi_data_used[k] for k in self.sdu_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.sdu_plot_data['X'])] * len(self.sdu_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.sdu_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' SDU per Epoch',
                    'legend': self.sdu_plot_data['legend'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Semi-Data Used (%)'},
                win=self.display_id+2)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_max_confidence(self, epoch, confidence):
        """

        Input params:
            epoch: Current epoch.
            semi_data_used: The percentage of semi_data used and stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'max_conf_plot_data'):
            self.max_conf_plot_data = {'X': [], 'Y': [], 'legend': list(confidence.keys())}
        self.max_conf_plot_data['X'].append(epoch)
        self.max_conf_plot_data['Y'].append([confidence[k] for k in self.max_conf_plot_data['legend']])
        # x = np.squeeze(np.stack([np.array(self.max_conf_plot_data['X'])] * len(self.max_conf_plot_data['legend']), 1), axis=1)
        # y = np.squeeze(np.array(self.max_conf_plot_data['Y']), axis=1)
        x = np.stack([np.array(self.max_conf_plot_data['X'])] * len(self.max_conf_plot_data['legend']), 1)
        y = np.array(self.max_conf_plot_data['Y'])
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' Max Confidence',
                    'legend': self.max_conf_plot_data['legend'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Confidence'},
                win=self.display_id+3)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_semi_classes(self, epoch, classes):
        """

        Input params:
            epoch: Current epoch.
            semi_data_used: The percentage of semi_data used and stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'semi_classes_plot_data'):
            self.semi_classes_plot_data = {'X': [], 'Y': [], 'legend': list(classes.keys())}
        self.semi_classes_plot_data['X'].append(epoch)
        self.semi_classes_plot_data['Y'].append([classes[k] for k in self.semi_classes_plot_data['legend']])
        # x = np.squeeze(np.stack([np.array(self.max_conf_plot_data['X'])] * len(self.max_conf_plot_data['legend']), 1), axis=1)
        # y = np.squeeze(np.array(self.max_conf_plot_data['Y']), axis=1)
        x = np.stack([np.array(self.semi_classes_plot_data['X'])] * len(self.semi_classes_plot_data['legend']), 1)
        y = np.array(self.semi_classes_plot_data['Y'])
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' Classes Count For Semi-Supervised',
                    'legend': self.semi_classes_plot_data['legend'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Number Used'},
                win=self.display_id+4)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_max_confidence_hist(self, epoch, hist):
        """

        Input params:
            epoch: Current epoch.
            semi_data_used: The percentage of semi_data used and stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'max_conf_hist_data'):
            self.max_conf_hist_data = {'X': [], 'Y': [], 'legend': list(hist.keys())}
        self.max_conf_hist_data['X'].append(epoch)
        self.max_conf_hist_data['Y'].append([hist[k] for k in self.max_conf_hist_data['legend']])
        # x = np.squeeze(np.stack([np.array(self.max_conf_plot_data['X'])] * len(self.max_conf_plot_data['legend']), 1), axis=1)
        # y = np.squeeze(np.array(self.max_conf_plot_data['Y']), axis=1)
        x = np.stack([np.array(self.max_conf_hist_data['X'])] * len(self.max_conf_hist_data['legend']), 1)
        y = np.array(self.max_conf_hist_data['Y'])
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' Max Confidence Count By Class',
                    'legend': self.max_conf_hist_data['legend'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Count'},
                win=self.display_id+5)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_class_errors(self, epoch, errors):
        """

        Input params:
            epoch: Current epoch.
            semi_data_used: The percentage of semi_data used and stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'class_error_plot_data'):
            self.class_error_plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.class_error_plot_data['X'].append(epoch)
        self.class_error_plot_data['Y'].append([errors[k] for k in self.class_error_plot_data['legend']])
        # x = np.squeeze(np.stack([np.array(self.max_conf_plot_data['X'])] * len(self.max_conf_plot_data['legend']), 1), axis=1)
        # y = np.squeeze(np.array(self.max_conf_plot_data['Y']), axis=1)
        x = np.stack([np.array(self.class_error_plot_data['X'])] * len(self.class_error_plot_data['legend']), 1)
        y = np.array(self.class_error_plot_data['Y'])
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' Classification Error Count',
                    'legend': self.class_error_plot_data['legend'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Count'},
                win=self.display_id+6)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_current_validation_metrics(self, epoch, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_roc_curve(self, fpr, tpr, thresholds):
        """Display the ROC curve.

        Input params:
            fpr: False positive rate (1 - specificity).
            tpr: True positive rate (sensitivity).
            thresholds: Thresholds for the curve.
        """
        try:
            self.vis.line(
                X=fpr,
                Y=tpr,
                opts={
                    'title': 'ROC Curve',
                    'xlabel': '1 - specificity',
                    'ylabel': 'sensitivity',
                    'fillarea': True},
                win=self.display_id+2)
        except ConnectionError:
            self.create_visdom_connections()


    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        # zip the images together so that always the image is followed by label is followed by prediction
        images = images.permute(1,0,2,3)
        images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3]))

        # add a channel dimension to the tensor since the excepted format by visdom is (B,C,H,W)
        images = images[:,None,:,:]

        try:
            self.vis.images(images, win=self.display_id+3, nrow=3)
        except ConnectionError:
            self.create_visdom_connections()


    def print_current_losses(self, epoch, max_epochs, iter, max_iters, losses):
        """Print current losses on console.

        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = '[epoch: {}/{}, iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
        for k, v in losses.items():
            message += '{0}: {1:.6f} '.format(k, v)

        print(message)  # print the message
