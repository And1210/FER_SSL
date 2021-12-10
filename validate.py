import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import numpy as np
from collections import OrderedDict

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(config_file, start_epoch, end_epoch):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    datasets = []
    dir_names = [os.path.join(configuration["model_params"]["checkpoint_path"], configuration[d]["dataset_name"]) for d in configuration["datasets"]]
    index = 0
    for d in configuration["datasets"]:
        print('Initializing dataset...')
        val_dataset = create_dataset(configuration[d])
        val_dataset_size = len(val_dataset)
        print('The number of validation samples = {0}'.format(val_dataset_size))
        print('Dataset {} loaded'.format(configuration[d]["dataset_name"]))
        datasets.append(val_dataset)
        dir_name = dir_names[index]
        if (not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        index += 1
    dataset_accuracy = [[] for i in range(len(datasets))]

    best_accuracy = [0 for i in range(len(datasets))]
    best_epoch = [0 for i in range(len(datasets))]

    if (start_epoch >= 0 and end_epoch >= 0):
        print('Initializing visualization...')
        visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots
        for epoch in range(start_epoch, end_epoch+1):
            configuration['model_params']['load_checkpoint'] = epoch
            model = create_model(configuration['model_params'])
            model.setup()
            model.eval()

            index = 0
            for d in datasets:
                model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])
                for i, data in enumerate(d):
                    model.set_input(data)  # unpack data from data loader
                    model.test()           # run inference

                dataset_accuracy[index].append(model.post_epoch_callback_validate(configuration['model_params']['load_checkpoint'], visualizer))
                if (dataset_accuracy[index][-1] > best_accuracy[index]):
                    best_accuracy[index] = dataset_accuracy[index][-1]
                    best_epoch[index] = epoch
                    model.save_dir = os.path.join(configuration["model_params"]["checkpoint_path"], dir_names[index])
                    print('New best found on epoch {} for dataset {}'.format(epoch, configuration[configuration["datasets"][index]]["dataset_name"]))
                    model.save_networks(epoch)
                    model.save_optimizers(epoch)
                index += 1
            metrics = OrderedDict()
            index = 0
            for i in configuration["datasets"]:
                metrics['{} Accuracy'.format(configuration[i]["dataset_name"])] = dataset_accuracy[index][-1]
                index += 1
            visualizer.plot_current_validation_metrics_multi(epoch, metrics)
    else:
        print('Initialzing model...')
        model = create_model(configuration['model_params'])
        model.setup()
        model.eval()

        print('Initializing visualization...')
        visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

        index = 0
        for d in datasets:
            model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])
            for i, data in enumerate(d):
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference

            dataset_accuracy[index].append(model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer))
    dataset_accuracy = np.array(dataset_accuracy)

    index = 0
    for d in datasets:
        print('Accuracy of dataset {}: {}'.format(configuration[configuration["datasets"][index]]["dataset_name"], dataset_accuracy[index]))
        index += 1
    print('Mean accuracy: {}'.format(np.mean(dataset_accuracy, axis=0)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')
    parser.add_argument('start_epoch', default=-1, help='starting epoch to test')
    parser.add_argument('end_epoch', default=-1, help='ending epoch to test(inclusive)')

    args = parser.parse_args()
    validate(args.configfile, int(args.start_epoch), int(args.end_epoch))
