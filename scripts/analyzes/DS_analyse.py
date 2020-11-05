import os
import sys
from os import path

import numpy as np

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pandas as pd
from sklearn import preprocessing


# load dataset
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


# retrieve unique values in all dimensions of a dataset -> that would be the 'alphabet' of possible events
def retrieve_unique_values(dataset):
    flat_dataset = dataset.values.flatten()
    unique_value = list(set(flat_dataset))
    # print(len(unique_value))
    # print(unique_value)
    return unique_value


# preprocessing dataset
def preprocessing_to_num(dataset):
    values = retrieve_unique_values(dataset)

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

    return encoded_events, event2int, int2event


# preprocessing categories
def preprocessing_cat(categories):
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(categories))

    print("Categories:")
    print(lb.classes_)
    cl_enc = lb.transform(categories)

    return lb, pd.DataFrame(cl_enc)


def analyse(cl_decoder, data, output):
    x_no_dup = data.drop_duplicates()
    cl_decod = cl_decoder.inverse_transform(output.values)

    xy = pd.concat([data, pd.DataFrame(cl_decod,
                                       columns=["out"])], axis=1, join='inner')
    xy_no_dup = xy.drop_duplicates()

    xy_only_dup = xy_no_dup.drop(x_no_dup.index)

    doublons = []
    nb_tr = dict()
    nb_tr_per_classes = dict()
    nb_tr_uniq_per_classes = dict()
    nb_tr_dup_per_classes = dict()
    for cle in cl_decod:
        nb_tr[cle] = 0
        nb_tr_per_classes[cle] = 0
        nb_tr_uniq_per_classes[cle] = 0
        nb_tr_dup_per_classes[cle] = 0

    for cle in cl_decod:
        nb_tr[cle] = nb_tr[cle] + 1

    for id1, t0 in xy_no_dup.iterrows():
        out1 = cl_decod[id1]
        nb_tr_per_classes[out1] = nb_tr_per_classes[out1] + 1

    for id2, t0 in x_no_dup.iterrows():
        t2 = data.loc[id2]
        out2 = cl_decod[id2]
        duplicates = [out2]
        for id3, t03 in xy_only_dup.iterrows():
            t3 = data.loc[id3]
            out3 = cl_decod[id3]
            if (t2.equals(t3)) & (out2 != out3):
                duplicates.append(out3)
        if len(duplicates) > 1:
            doublons.append(duplicates)
            for cl in duplicates:
                nb_tr_dup_per_classes[cl] = nb_tr_dup_per_classes[cl] + 1
        else:
            nb_tr_uniq_per_classes[out2] = nb_tr_uniq_per_classes[out2] + 1

    print("Nombre de traces totales: ", len(data))
    print("Nombre de traces uniques (sortie comprise): ", len(xy_no_dup))
    print("Nombre de traces uniques (sortie non comprise): ", len(x_no_dup))
    print("Liste des doublons", doublons)
    print("Nombre de traces par classes: ", nb_tr)
    print("Nombre de traces uniques par classes: ", nb_tr_per_classes)
    prct_tr_per_classes = {k: round((v / nb_tr[k]), 5) * 100 for k, v in nb_tr_per_classes.items()}
    print("% de traces uniques par classes: ", prct_tr_per_classes)
    print("Nombre de traces uniquement classifiées dans cette classe: ", nb_tr_uniq_per_classes)
    nb_tr_uniq_per_classes = {k: round((v / nb_tr_per_classes[k]), 5) * 100 for k, v in nb_tr_uniq_per_classes.items()}
    print("% de traces uniquement classifiées dans cette classe: ", nb_tr_uniq_per_classes)
    print("Nombre de traces classifiées à la fois dans cette classes et dans minimum une autre classe: ",
          nb_tr_dup_per_classes)
    nb_tr_dup_per_classes = {k: round((v / nb_tr_per_classes[k]), 5) * 100 for k, v in nb_tr_dup_per_classes.items()}
    print("% de traces classifiées à la fois dans cette classes et dans minimum une autre classe: ",
          nb_tr_dup_per_classes)


def main(dataset_filename):
    print(dataset_filename)

    df_dataset = load_dataset(dataset_filename)

    classes = df_dataset['Category']
    del df_dataset['Category']

    cl_decod, cl_encoded = preprocessing_cat(list(classes))

    encoded_ev, event2int, int2event = preprocessing_to_num(df_dataset)

    # just in case
    df_encoded_ev = pd.DataFrame(encoded_ev)

    output_directory = "../../results/DS_analyze/"
    output_filename_base = path.basename(dataset_filename)
    output_filename = output_filename_base + '.txt'

    output_file = output_directory + output_filename
    f = open(output_file, "a")
    orig_stdout = sys.stdout
    sys.stdout = f

    analyse(cl_decod, df_encoded_ev, cl_encoded)

    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    dataset_filename = 'HospitalBilling.csv'
    # dataset_filename = 'BPIC15.csv'
    # main(dataset_filename)

    # dataset_filename = 'BPIC20.csv'
    main(dataset_filename)
