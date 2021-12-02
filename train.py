import argparse
from datasets import create_dataset
from utils import parse_configuration
import math
import random
from models import create_model
import time
from utils.visualizer import Visualizer
from collections import OrderedDict
import numpy as np
import torch

"""Performs training of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=True):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    train_dataset = create_dataset(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))

    if (configuration["model_params"]["use_semi"]):
        semi_dataset = create_dataset(configuration['semi_dataset_params'])
        semi_dataset_size = len(semi_dataset)
        print('The number of semi-supervised samples = {0}'.format(semi_dataset_size))

    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots

    starting_epoch = configuration['model_params']['load_checkpoint'] + 1
    num_epochs = configuration['model_params']['max_epochs']

    batch_size = configuration["train_dataset_params"]["loader_params"]["batch_size"]
    semi_percentage = configuration["model_params"]["semi_percentage"]
    if (not configuration["model_params"]["use_semi"]):
        semi_percentage = 1
    else:
        class_labels = [[] for i in range(7)]
        for i, data in enumerate(train_dataset):
            labels = data[1].cpu().detach().numpy()
            count = 0
            for label in labels:
                class_labels[label].append({'class': label, 'use_data': random.random() < semi_percentage, 'index': i*16+count})
                count += 1

    # if (configuration['model_params']['load_checkpoint'] < 0):
    #     class_labels_list = []
    #     for c in class_labels:
    #         class_labels_list += c
    #     class_labels_list = sorted(class_labels_list, key=lambda x: x['index'])
    #     percentage_used_by_class = [0 for i in range(len(class_labels))]
    #     for c in class_labels_list:
    #         if (c['use_data']):
    #             percentage_used_by_class[c['class']] += 1
    #     labels_count = [len(class_labels[i]) for i in range(len(percentage_used_by_class))]
    #     for i in range(len(labels_count)):
    #         if (labels_count[i] <= 0):
    #             labels_count[i] = 1
    #     percentage_used_by_class = [percentage_used_by_class[i]/labels_count[i] for i in range(len(percentage_used_by_class))]
    #     print(percentage_used_by_class)
    #     # use_data = [True if random.random() < semi_percentage else False for i in range(len(train_dataset))]
    #     use_data = [c['use_data'] for c in class_labels_list]
    #     print(len(use_data))
    # else:
    #     filename = '{}_use_data_{}.csv'.format(configuration['model_params']['load_checkpoint'], 'model')
    #     print("Loading use_data array from {}".format(filename))
    #     use_data = np.loadtxt(filename)
    #     use_data = [d >= 0.5 for d in use_data]

    # labels = [data[1] for i, data in enumerate(train_dataset)]


    # counts = [0 for i in range(7)]
    # for i, data in enumerate(train_dataset):
    #     for j in range(batch_size):
    #         try:
    #             counts[data[1][j]] += 1
    #         except:
    #             break
    # print("Class counts: {}".format(counts))

    if (configuration["model_params"]["use_semi"]):
        dataset_ratio = len(semi_dataset)/(len(semi_dataset)+len(train_dataset))
        if (configuration['model_params']['load_checkpoint'] < 0):
            use_data = [False for i in range(len(semi_dataset))]
        else:
            filename = '{}_use_data_{}.csv'.format(configuration['model_params']['load_checkpoint'], 'model')
            print("Loading use_data array from {}".format(filename))
            use_data = np.loadtxt(filename)
            use_data = [d >= 0.5 for d in use_data]
        thresholds = [-1 for i in range(7)]
        labels = semi_dataset.dataset.labels
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
        semi_batch_size = configuration['semi_dataset_params']['loader_params']['batch_size']

        total_loss = 0
        iterations = 0

        semi_labels_correct = 0
        semi_labels = 0

        max_confidence = [-1 for i in range(7)]
        max_confidence_histogram = [0 for i in range(7)]
        classes_errors = [0 for i in range(7)]
        classes_updated = [0 for i in range(7)]

        if (configuration["model_params"]["use_semi"]):
            if (epoch >= configuration["model_params"]["start_semi_epoch"]):
                model.eval()
                for j in range(len(semi_dataset)):
                    if (not use_data[j]):
                        data = semi_dataset.dataset[j]

                        cur_data = torch.unsqueeze(data[0], 0)
                        model.set_input([cur_data, torch.tensor(labels[j])])
                        output = model.forward()
                        confidence = output
                        confidence = confidence.cpu().detach().numpy()[0]
                        cur_label = np.argmax(confidence)

                        max_confidence_histogram[cur_label] += 1
                        if (confidence[cur_label] > max_confidence[cur_label]):
                            max_confidence[cur_label] = confidence[cur_label]
                        thresh = thresholds[cur_label]
                        if (confidence[cur_label] >= thresh and confidence[cur_label] >= configuration["model_params"]["semi_thresh"]):
                            use_data[j] = True
                            labels[j] = cur_label

        print(max_confidence)
        thresholds = [2*((c/2+0.5)*configuration["model_params"]["rel_semi_thresh"]-0.5) for c in max_confidence]
        print(thresholds)

        used_semi = False
        used_train = False

        if (not configuration["model_params"]["use_semi"] or sum(use_data) == 0 or epoch < configuration["model_params"]["start_semi_epoch"]):
            used_semi = True

        if (epoch >= configuration["model_params"]["start_semi_epoch"] and configuration["model_params"]["use_semi_only"]):
            used_train = True

        semi_index = 0
        train_index = 0
        i = 0
        while (not used_semi or not used_train):
            model.train()
            data_queue = [[], []]
            while (len(data_queue[0]) < train_batch_size):
                if (configuration["model_params"]["use_semi"]):
                    if (semi_index >= len(semi_dataset)):
                        semi_index = 0
                        used_semi = True
                    if (train_index >= len(train_dataset)):
                        train_index = 0
                        used_train = True
                    if (not configuration["model_params"]["use_semi_only"]):
                        if (sum(use_data) > 0 and random.random() < dataset_ratio):
                            while (not use_data[semi_index]):
                                semi_index += 1
                                if semi_index >= len(semi_dataset):
                                    semi_index = 0
                                    used_semi = True
                            data_queue[0].append(semi_dataset.dataset[semi_index][0])
                            data_queue[1].append(torch.tensor(labels[semi_index]))
                            semi_index += 1
                        else:
                            data_queue[0].append(train_dataset.dataset[train_index][0])
                            data_queue[1].append(torch.tensor(train_dataset.dataset[train_index][1]))
                            train_index += 1
                    else:
                        if (epoch >= configuration["model_params"]["start_semi_epoch"]):
                            while (not use_data[semi_index]):
                                semi_index += 1
                                if semi_index >= len(semi_dataset):
                                    semi_index = 0
                                    used_semi = True
                            data_queue[0].append(semi_dataset.dataset[semi_index][0])
                            data_queue[1].append(torch.tensor(labels[semi_index]))
                            semi_index += 1
                        else:
                            data_queue[0].append(train_dataset.dataset[train_index][0])
                            data_queue[1].append(torch.tensor(train_dataset.dataset[train_index][1]))
                            train_index += 1
                else:
                    if (train_index >= len(train_dataset)):
                        train_index = 0
                        used_train = True
                    data_queue[0].append(train_dataset.dataset[train_index][0])
                    data_queue[1].append(torch.tensor(train_dataset.dataset[train_index][1]))
                    train_index += 1

            input_data = [torch.stack(data_queue[0]), torch.stack(data_queue[1])]

            visualizer.reset()

            model.set_input(input_data)         # unpack data from dataset and apply preprocessing
            output = model.forward()
            model.compute_loss()

            total_loss += model.loss_total.item()

            if i % configuration['model_update_freq'] == 0:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if i % configuration['printout_freq'] == 0:
                if (configuration["model_params"]["use_semi"]):
                    total_iterations = train_iterations + sum(use_data)
                else:
                    total_iterations = train_iterations
                if (configuration["model_params"]["use_semi_only"] and epoch >= configuration["model_params"]["start_semi_epoch"]):
                    total_iterations -= train_iterations
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, num_epochs, i, math.floor(total_iterations / train_batch_size), losses)
                visualizer.plot_current_losses(epoch, float(i) / math.floor(total_iterations / train_batch_size), losses)

            i += 1

        prev_max_confidence = max_confidence
        print("Max Confidence from Epoch {}".format(max_confidence))
        data = OrderedDict()
        for i in range(len(max_confidence)):
            data['Max Confidence {}'.format(i)] = max_confidence[i]
        visualizer.plot_max_confidence(epoch, data)
        if (configuration["model_params"]["use_semi"]):
            # print("Classes Updated From Semi-Supervised Learning: {}".format(classes_updated))
            # data = OrderedDict()
            # for i in range(len(classes_updated)):
            #     data['Class {}'.format(i)] = classes_updated[i]
            # visualizer.plot_semi_classes(epoch, data)
            data_used = 100*sum(use_data)/len(semi_dataset)
            print("Percentage of used data: {:.2f}%".format(data_used))
            # if (semi_labels > 0):
            #     print("Semi-Supervised Labelling Percentage Correct: {:.2f}%".format(100*semi_labels_correct/semi_labels))
            #     data = OrderedDict() #data to send to plot
            #     data['ArgMax Labelling % Correct'] = 100*semi_labels_correct/semi_labels
            # else:
            #     print("Semi-Supervised Labelling Percentage Correct: 0%")
            #     data = OrderedDict()
            #     data['ArgMax Labelling % Correct'] = 0
            # visualizer.plot_current_semi_data_used(epoch, data)
            # print("Semi-Supervised Labelling Done: {}".format(semi_labels))

        # print('Loss {}\nData Len {}'.format(total_loss, len(train_dataset)))
        # print('Loss for epoch {}: {}'.format(epoch, total_loss/float(iterations)))
        model.eval()
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            model.test()

        model.post_epoch_callback(epoch, visualizer)
        train_dataset.dataset.post_epoch_callback(epoch)

        print('Saving model at the end of epoch {0}'.format(epoch))
        model.save_networks(epoch)
        model.save_optimizers(epoch)
        if (configuration["model_params"]["use_semi"]):
            np.savetxt('{}_use_data_{}.csv'.format(epoch, 'model'), use_data)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))

        model.update_learning_rate() # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_configuration = configuration['train_dataset_params']
        custom_configuration['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = train_dataset.get_custom_dataloader(custom_configuration)
        sample_input = next(iter(dl)) # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', default="./config_fer.json", help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
