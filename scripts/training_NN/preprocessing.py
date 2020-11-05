import numpy as np
import sys
from os import path
from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pandas as pd
from sklearn import preprocessing


# preprocessing to a multi-labelled dataset
def multi_labelled(file_name, data):
    dir = '../../datasets/multi-label/'
    file = dir + '/' + file_name
    classes_single = data.filter(regex="Category.*")
    x = data.drop(classes_single.columns, axis=1)
    x_no_dup = x.drop_duplicates()
    xy_no_dup = data.drop_duplicates()
    xy_only_dup = xy_no_dup.drop(x_no_dup.index)
    cl_decod, df_cl_encoded = preprocessing_cat(classes_single)
    classes_multi = pd.DataFrame()

    for ind, tr in x_no_dup.iterrows():
        out = df_cl_encoded.loc[ind]
        result = out
        for ind2, y2 in xy_only_dup.iterrows():
            tr2 = x.loc[ind2]
            out2 = df_cl_encoded.loc[ind2]
            if tr.equals(other=tr2) & (not (out.equals(other=out2))):
                result = result.add(out2)
        classes_multi = classes_multi.append(result, ignore_index=True)

    cl_names = cl_decod.inverse_transform(np.diag(classes_multi.columns.values))
    classes_multi.columns = cl_names
    classes_multi = classes_multi.add_prefix("Category_")
    data_multi = classes_multi.join(x_no_dup.reset_index(drop=True))
    data_multi.to_csv(file, encoding='utf-8', index=False)


# load dataset
def load_dataset(dataset_filename, multi_label):
    # load dataset .csv according to first argument
    if multi_label:
        dataset_dir = '../../datasets/multi-label/'
    else:
        dataset_dir = '../../datasets/single-label/'

    file_to_load = dataset_dir + dataset_filename
    print(f"Trying to load: {file_to_load}")
    if not path.exists(dataset_dir):
        print(f"The expected dataset directory {dataset_dir} was not found.  Multi-labelling is: {multi_label}")
        sys.exit()
    elif (not path.exists(str(dataset_dir + '/' + dataset_filename))) & multi_label:
        print("There is no dataset for multi-labelled data. Trying to generate it from a single-labelled "
              "corresponding dataset...")
        single_dataset = load_dataset(dataset_filename, False)
        print("Single-labelled dataset found. Creating the multi-labelled version...")
        multi_labelled(dataset_filename, single_dataset)
        print("File " + file_to_load + " created successfully.")
    elif not path.exists(file_to_load):
        print(file_to_load)
        print("The dataset file was not found")
        sys.exit()

    # dealing with process of different lengths by filling missing values
    dataset = np.genfromtxt(file_to_load, delimiter=',', missing_values='', filling_values='', skip_header=0,
                            names=True, dtype=None, encoding="utf-8")
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


# preprocessing
def preproc(dataset_filename, multi):
    df_dataset = load_dataset(dataset_filename, multi)
    classes = df_dataset.filter(regex="Category.*")
    df_dataset = df_dataset.drop(classes.columns, axis=1)
    df_cl_decod, df_cl_encoded = preprocessing_cat(classes)
    encoded_ev, event2int, int2event = preprocessing_to_num(df_dataset)
    df_ev_encoded = pd.DataFrame(encoded_ev)
    return df_ev_encoded, df_cl_encoded, event2int, int2event

# Pas sure que ce soit nécessaire...
# disable eager execution
# tf.compat.v1.disable_eager_execution()

# mlb = MultiLabelBinarizer() instead of LabelBinarizer() -> Pas besoin car fait manuellement
