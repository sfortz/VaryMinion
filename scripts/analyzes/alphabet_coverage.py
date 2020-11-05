import sys
from os import path

import numpy as np
import pandas as pd


def load_dataset(dataset_filename):
    # load dataset .csv according to first argument
    dataset_dir = '../../datasets/'
    if not path.exists(dataset_dir):
        print("The expected dataset directory was not found")
        sys.exit()
    elif not path.exists(str(dataset_dir + '/' + dataset_filename)):
        print(str(dataset_dir + '/' + dataset_filename))
        print("The dataset file was not found")
        sys.exit()
    file_to_load = dataset_dir + '/' + dataset_filename
    # dealing with process of different lengths by filling missing values
    dataset = np.genfromtxt(file_to_load, delimiter=',', missing_values='', filling_values='', skip_header=0,
                            names=True, dtype=None)
    pd_dataset = pd.DataFrame(dataset)
    return pd_dataset


def retrieve_unique_values(dataset):
    flat_dataset = dataset.values.flatten()
    unique_value = list(set(flat_dataset))
    # print(len(unique_value))
    # print(unique_value)
    return unique_value


if __name__ == "__main__":
    dataset_filename = ""
    percent_training = 0.66
    nb_iter = 11
    # dealing with arguments: argv[1] is the filename of the dataset; argv[2] is the percentage of data in the training set
    argc = len(sys.argv)

    if (argc >= 2):
        dataset_filename = sys.argv[1]
    else:
        dataset_filename = 'BPIC15.csv'

    if (argc > 2):
        percent_training = float(sys.argv[2])
        # can give both 0.66 and 66 for instance
        if percent_training > 1.0:
            percent_training = percent_training / 100

    if (argc > 3):
        nb_iter = int(sys.argv[3]) + 1

    dataset = load_dataset(dataset_filename)
    print(dataset.head())
    print(dataset.shape)

    classes = dataset['Category']
    del dataset['Category']

    values = retrieve_unique_values(dataset)
    nb_unique_values = len(values)

    # Creating a dictionary that maps integers to the events/actions (adapted from https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)
    int2event = dict(enumerate(values))
    # Creating another dictionary that maps events/actions to integers (adapted from https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)
    event2int = {char: ind for ind, char in int2event.items()}
    # print(event2int)

    # encoding events with mapping dictionary
    encoded_events = []
    # print(range(len(dataset)))
    for i, row in dataset.iterrows():
        encoded_events.append(row)

    for i, row in dataset.iterrows():
        encoded_events[i] = [event2int[event] for event in row]

    # print(encoded_events)

    # are all classes equally represented?
    print(classes.value_counts(dropna=False))

    # compute an accumulator of frequency or each event -> check that missing events in the training (if any) are rare
    accumul_events = np.zeros(nb_unique_values, dtype=int)
    # print(accumul_events)
    print("size of the accumulator")
    print(accumul_events.shape)

    for i, row in dataset.iterrows():
        for event in row:
            accumul_events[event2int[event]] += 1

    print(accumul_events)

    # create two sets
    import random

    dataset_init = dataset
    classes_init = classes

    output_directory = '../../results/alphabet_coverage/'
    output_filename_base = path.basename(dataset_filename)

    nb_l = dataset.shape[0]

    for it in range(1, nb_iter):
        dataset = dataset_init
        classes = classes_init

        # 66% of data for training set
        training_set_idx = random.sample(range(nb_l), int(nb_l * percent_training))
        tr_X = dataset.iloc[training_set_idx]
        tr_Y = classes.iloc[training_set_idx]

        ts_X = dataset.drop(training_set_idx)
        ts_Y = classes.drop(training_set_idx)

        print("Saving training et test sets")

        df = pd.concat([tr_X, tr_Y], axis=1)
        df.to_csv(output_directory + output_filename_base + '_trSet_iter' + str(it) + '.csv')

        df = pd.concat([ts_X, ts_Y], axis=1)
        df.to_csv(output_directory + output_filename_base + '_tsSet_iter' + str(it) + '.csv')

        output_filename = output_filename_base + '_output_iter' + str(it) + '.txt'
        output_file = output_directory + output_filename

        f = open(output_file, "w")
        orig_stdout = sys.stdout
        sys.stdout = f

        # count number of events in training set
        unique_event_trX = retrieve_unique_values(tr_X)
        encoded_event_trX = [event2int[event] for event in unique_event_trX]
        print(len(encoded_event_trX))
        print(encoded_event_trX)

        print("Number of instances per classes in the training set")
        print(tr_Y.value_counts(dropna=False))

        print("Are missing events rare?")
        set_events = set(range(nb_unique_values))
        set_events_in_training = set(encoded_event_trX)
        missing_events = set_events - set_events_in_training
        print(nb_unique_values - len(encoded_event_trX))
        print(missing_events)
        for val in missing_events:
            print(accumul_events[val])
        # missing_events = [x for x in l_events if x not in encoded_event_trX]
        # print(event2int[x for x in missing_events])

        sys.stdout = orig_stdout
        f.close()
