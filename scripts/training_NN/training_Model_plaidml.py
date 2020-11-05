import argparse
import sys
import time
from os import path
from os import environ
import numpy as np
import numpy.ma as ma

environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # NB:  PlaidML backend has to be imported before keras to work
# (sorry PEP8 ;))

import preprocessing
import training_GRU_plaidml
import training_LSTM_plaidml
import training_RNN_plaidml
from sklearn.model_selection import train_test_split
import plaidml
import plaidml.settings
import json

PLAIDML_IDS = ''  # store the name of CPU/GPU used by plaidML (configure it through plaidml-setup)

# getting information on the plaidML device used ...
def get_device_information():
    """ This function determines the type of processor (CPU/GPU) used by PlaidML """
    global PLAIDML_IDS
    with open(plaidml.settings.user_settings) as f:
        data = json.load(f)
    dev_info = str(data.get('PLAIDML_DEVICE_IDS')[0])

    if 'cpu' in  dev_info:
        print("Running PlaidML on CPU " + dev_info)
        PLAIDML_IDS = 'CPU_' + dev_info
    else:
        print("Running PlaidML on CPU " + dev_info)
        PLAIDML_IDS = 'GPU_' + dev_info




# create a neural network
# adapted from https://keras.io/guides/working_with_rnns/
def get_compiled_model(model_type="RNN", multi=False, alpha_size=128, nb_classes=1, nb_col=128, nb_unit=10):
    if model_type == "RNN":
        print("Training a RNN")
        model = training_RNN_plaidml.get_RNN_model(multi, alpha_size, nb_classes, nb_col, nb_unit)

    elif model_type == "LSTM":
        print("Training a LSTM")
        model = training_LSTM_plaidml.get_LSTM_model(multi, alpha_size, nb_classes, nb_col, nb_unit)

    elif model_type == "GRU":
        print("Training a GRU")
        model = training_GRU_plaidml.get_GRU_model(multi, alpha_size, nb_classes, nb_col, nb_unit)
    else:
        sys.exit("ERROR: " + model_type + " is not recognized as a model type")

    return model


def analyze_predictions(multi, predictions, np_class):
    samples = predictions[:15]

    if multi:
        for i in range(0, len(samples)):  # for multi-label we look at the highest predictions
            print(" ============ " + "Sample: " + str(i) + " ============")
            pred = (-samples[i]).argsort()  # predictions[i]
            print("pred sorted: " + str(pred) + " pred: " + str(samples[i]))
            np_masked_class = ma.masked_equal(np_class[i], 0)
            # print("masked class: " + str(np_masked_class))
            label = np_masked_class.nonzero()
            # print("label sorted: " + str(label[0]) + " vs " + str(tfclass[i]) + " for sample: " + str(i))
            print(" Top predictions indices: " + str(pred[:len(label[0])]) + " vs. real classes indices: " + str(label[0]))
            intersect = np.intersect1d(pred[:len(label[0])], label[0])
            union = np.union1d(pred[:len(label[0])], label[0])
            jaccard_score = len(intersect) / len(union)
            print(f"Jaccard Score: {jaccard_score}")
            print(" ")
    else:  # for single-label, we only need to retrieve the index of the maximum probability value corresponding to
        # the index of the selected class in the ground truth
        pred = np.argmax(predictions, axis=1)[:15]
        label = np.argmax(np_class, axis=1)[:15]
        # print(tftest[:15])
        # print(tfclass[:15])
        print(predictions[:15])
        print("Prediction:      " + str(pred))
        print("Expected outcome:" + str(label))


def main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training):
    start_time = time.time()
    print(dataset_filename)
    print(model_type)
    get_device_information()

    # we can give both 0.66 and 66 for instance
    if percent_training > 1.0:
        percent_training = percent_training / 100

    ev_encoded, cl_encoded, event2int, int2event = preprocessing.preproc(dataset_filename, multi)

    x_tr, x_ts, y_tr, y_ts = train_test_split(ev_encoded, cl_encoded, train_size=percent_training)

    print("Shapes of split data")
    # print(y_tr)
    print("x_tr training input shape: {0}".format(x_tr.shape))
    print("y_tr training output shape: {0}".format(y_tr.shape))
    print("x_ts test input shape: {0}".format(x_ts.shape))
    print("y_ts test output shape: {0}".format(y_ts.shape))

    # Our vectorized labels

    print("Internal unit: hypothesis to 10")
    print("Pandas:")
    print(x_tr.iloc[0])
    print(y_tr.iloc[0])

    print("Alphabet size:")
    print(len(event2int))

    model = get_compiled_model(model_type=model_type, multi=multi, alpha_size=len(event2int),
                               nb_classes=len(cl_encoded.columns),
                               nb_col=len(ev_encoded.columns), nb_unit=nb_unit)

    # print model characteristics
    print(model.summary())

    # converting training/test sets into numpy for improved GPU performance...
    np_xtrain = x_tr.to_numpy()
    np_xtest = x_ts.to_numpy()
    np_ytrain = y_tr.to_numpy()
    np_ytest = y_ts.to_numpy()

    model.fit(np_xtrain, np_ytrain, epochs=nb_epochs, batch_size=batch_size)

    print("Evaluate on test data")
    results = model.evaluate(np_xtest, np_ytest, batch_size=batch_size)
    print("Generating predictions")
    predictions = model.predict(np_xtest)
    print("Analyzing predictions")
    analyze_predictions(multi, predictions, np_ytest)


    output_directory = "../../results/training_metrics/"
    output_filename_base = path.basename(dataset_filename)
    output_filename = output_filename_base + '_metrics_' + str(model_type) + '_nb_unit_' + str(
        nb_unit) + '_training_set_size_' + str(
        percent_training) + '_nb_epochs_' + str(nb_epochs) + '_batch_size_' + str(batch_size) + \
        '_' + PLAIDML_IDS + '_PlaidML' + '.txt'

    output_file = output_directory + output_filename
    f = open(output_file, "a")
    orig_stdout = sys.stdout
    sys.stdout = f

    seconds = time.time() - start_time
    exec_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
    results.append(exec_time)

    print("test loss, test acc, exec time:", results)

    sys.stdout = orig_stdout
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VariabilityMiningClassifier V0.1')
    parser.add_argument('--dataset', default='BPIC15.csv', type=str, dest="dataset_filename",
                        help="Name of the dataset (default: BPIC15.csv)")
    parser.add_argument('--model_type', default="RNN", dest="model_type",
                        help="Choose amongst: RNN, LTSM or GRU (default: RNN")
    parser.add_argument('--multi', default=False, dest="multi",
                        help="Enable option to choose multi-labelling instead of single-labelling", action="store_true")
    parser.add_argument('--nb_epochs', default="50", dest="nb_epochs", type=int, help="Number of epochs (default=50)")
    parser.add_argument('--nb_units', default="10", dest="nb_unit", type=int,
                        help="Number of units for the model (default=10)")
    parser.add_argument('--training_ratio', default="0.66", dest="percent_training", type=float,
                        help="Ratio of the dataset used for training (default=0.66)")
    parser.add_argument('--batch_size', default="128", dest="batch_size",
                        help="Size of the batch for training (default=128)", type=int)

    args = parser.parse_args()
    dataset_filename = args.dataset_filename
    percent_training = args.percent_training
    nb_epochs = args.nb_epochs
    nb_unit = args.nb_unit
    batch_size = args.batch_size
    model_type = args.model_type
    multi = args.multi


    main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training)
