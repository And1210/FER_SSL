import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer

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

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    if (start_epoch >= 0 and end_epoch >= 0):
        print('Initializing visualization...')
        visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots
        for epoch in range(start_epoch, end_epoch+1):
            configuration['model_params']['load_checkpoint'] = epoch
            model = create_model(configuration['model_params'])
            model.setup()
            model.eval()
    
            model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])
    
            for i, data in enumerate(val_dataset):
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference

            model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)
    else:
        print('Initialzing model...')
        model = create_model(configuration['model_params'])
        model.setup()
        model.eval()

        print('Initializing visualization...')
        visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

        model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

        for i, data in enumerate(val_dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference

        model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')
    parser.add_argument('start_epoch', default=-1, help='starting epoch to test')
    parser.add_argument('end_epoch', default=-1, help='ending epoch to test(inclusive)')

    args = parser.parse_args()
    validate(args.configfile, int(args.start_epoch), int(args.end_epoch))
