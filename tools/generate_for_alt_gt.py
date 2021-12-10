import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import numpy as np
import torch

BASE_EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
BASE_EMOTION_DICT_INVERSE = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}

"""Generate psuedo-labels ffrom a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def pseudo_label(config_file):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    #print('Loading training data for pseudo-label accuracy evaluation...')
    #train_dataset = create_dataset(configuration['train_dataset_params'])
    #train_dataset_size = len(train_dataset)
    #print('The number of training samples = {0}'.format(train_dataset_size))
    #print('Loading semi-supervised training data...')
    semi_dataset = create_dataset(configuration['semi_dataset_params'])
    semi_dataset_size = len(semi_dataset)
    print('The number of semi-supervised samples = {0}'.format(semi_dataset_size))    

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    #print('Initializing visualization...')
    #visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    max_confidence = [-1 for i in range(7)]
    labels = [l for l in semi_dataset.dataset.labels]
    use_data = [False for i in range(semi_dataset_size)]

    print('Gathering max confidences for relative thresholding...')
    model.eval()
    for j in range(len(semi_dataset)):
        data = semi_dataset.dataset[j]

        cur_data = torch.unsqueeze(data[0], 0)
        model.set_input([cur_data, torch.tensor(labels[j])])
        output = model.forward()
        confidence = output
        confidence = confidence.cpu().detach().numpy()[0]
        cur_label = np.argmax(confidence)
                    
        if (confidence[cur_label] > max_confidence[cur_label]):
            max_confidence[cur_label] = confidence[cur_label]

    thresholds = [2*((c/2+0.5)*configuration["model_params"]["rel_semi_thresh"]-0.5) for c in max_confidence]

    print('Computing use_data and labels arrays...')
    for j in range(len(semi_dataset)):
        if (not use_data[j]):
            data = semi_dataset.dataset[j]

            cur_data = torch.unsqueeze(data[0], 0)
            model.set_input([cur_data, torch.tensor(labels[j])])
            output = model.forward()
            confidence = output
            confidence = confidence.cpu().detach().numpy()[0]
            cur_label = np.argmax(confidence)
            emotion = semi_dataset.dataset.get_emotion(semi_dataset.dataset.labels[j])
            emotion_label = BASE_EMOTION_DICT_INVERSE[emotion]
            labels[j] = emotion_label
            use_data[j] = True

    np.savetxt(os.path.join(configuration["model_params"]["semi_data_output_path"], '{}_use_data_{}.csv'.format(0, 'model')), use_data)
    print("Saved use_data array at {}".format(configuration["model_params"]["semi_data_output_path"]))
    np.savetxt(os.path.join(configuration["model_params"]["semi_data_output_path"], '{}_labels_{}.csv'.format(0, 'model')), labels)
    print("Saved labels array at {}".format(configuration["model_params"]["semi_data_output_path"]))
    np.savetxt(os.path.join(configuration["model_params"]["semi_data_output_path"], '{}_max_confidence_{}.csv'.format(0, 'model')), max_confidence)
    print("Save max_confidence array at {}".format(configuration["model_params"]["semi_data_output_path"]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()
    pseudo_label(args.configfile)
