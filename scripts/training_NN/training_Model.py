import argparse
import sys
import time
from os import path
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow import test as tf_test
from tensorflow.keras.callbacks import TerminateOnNaN
from sklearn.model_selection import train_test_split
import preprocessing
import training_GRU
import training_LSTM
import training_RNN


TENSORFLOW_DEVICE = ''  # store if the type of computation is CPU or GPU


# FIXME: get a more precise characterisation of the CPU device used...
def get_tf_device_type():
    """ This function determines the type of processor (CPU or GPU) used by tensorflow """
    global TENSORFLOW_DEVICE
    if tf_test.is_gpu_available():
        print('Running Tensforflow with GPU, yay !')
        TENSORFLOW_DEVICE = tf_test.gpu_device_name()
    else:
        print('Running Tensforflow with CPU')
        TENSORFLOW_DEVICE = 'CPU'


# create a neural network
# adapted from https://keras.io/guides/working_with_rnns/
def get_compiled_model(model_type="RNN", multi=False, alpha_size=128, nb_classes=1, nb_col=128, nb_unit=10, activation='tanh', loss='mse'):
    # the model takes as input and output only one kind of such entities (Sequential) -> takes a list of Events and
    # outputs the configuration to which the trace belongs. In between, only the result of the neuron functions are
    # aggregated. model = tf.keras.Sequential([ tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(
    # 10, activation='relu'), tf.keras.layers.Dense(nb_classes) ])
    if model_type == "RNN":
        print("Training a RNN")
        model = training_RNN.get_RNN_model(multi, alpha_size, nb_classes, nb_col, nb_unit, activation, loss)
    elif model_type == "LSTM":
        print("Training a LSTM")
        model = training_LSTM.get_LSTM_model(multi, alpha_size, nb_classes, nb_col, nb_unit, activation, loss)
    elif model_type == "GRU":
        print("Training a GRU")
        model = training_GRU.get_GRU_model(multi, alpha_size, nb_classes, nb_col, nb_unit, activation, loss)
    else:
        sys.exit("ERROR: " + model_type + " is not recognized as a model type")

    return model


def analyze_predictions(multi, predictions, tfclass):
    samples = predictions[:15]
    np_class = tfclass.numpy()

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
        label = np.argmax(tfclass, axis=1)[:15]
        # print(tftest[:15])
        # print(tfclass[:15])
        print(predictions[:15])
        print("Prediction:      " + str(pred))
        print("Expected outcome:" + str(label))



def main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training, activation, loss):
    start_time = time.time()
    print(dataset_filename)
    get_tf_device_type()
    print(model_type)

    # we can give both 0.66 and 66 for instance
    if percent_training > 1.0:
        percent_training = percent_training / 100

    ev_encoded, cl_encoded, event2int, int2event = preprocessing.preproc(dataset_filename, multi)


    x_tr, x_ts, y_tr, y_ts = train_test_split(ev_encoded, cl_encoded, train_size=percent_training)

    print("output y_tr")
    # print(y_tr)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_ts.shape)
    print(y_ts.shape)

    # Our vectorized labels

    print("Test generation tensorFlow datasets")

    # turn into tensorFlow dataset
    # tf_train = tf.data.Dataset.from_tensor_slices((x_tr.values, y_tr.values))
    tf_train = tf.convert_to_tensor(x_tr)
    tf_label = tf.convert_to_tensor(y_tr)
    # for element in tf_train:
    #    print(element)
    # tf_test = tf.data.Dataset.from_tensor_slices((x_ts.values, y_ts.values))
    tf_test = tf.convert_to_tensor(x_ts)
    tf_class = tf.convert_to_tensor(y_ts)

    # reshape for tensorflow/keras RNN -> df_dataset.columns = number of features; one class to retrieve
    # tf_train = tf.reshape(tf_train,[-1,1,df_dataset.columns])

    # tf_test = tf.reshape(tf_test,[-1,1,df_dataset.columns])

    print("End generating tensorFlow datasets")

    print("Alphabet size:")
    print(len(event2int))
    model = get_compiled_model(model_type=model_type, multi=multi, alpha_size=len(event2int),
                               nb_classes=len(cl_encoded.columns),
                               nb_col=len(ev_encoded.columns), nb_unit=nb_unit, activation=activation, loss=loss)

    print(model.summary())

    callbacks = [TerminateOnNaN()]

    history = model.fit(tf_train, tf_label, epochs=nb_epochs, batch_size=batch_size, callbacks=callbacks)

    #print('Last loss value:')
    #print(list(history.history['loss'])[-1])

    print("Evaluate on test data")
    results = model.evaluate(tf_test, tf_class, batch_size=batch_size)

    print("Generate predictions")
    pred_noarg = model.predict(tf_test)
    print("Analyzing predictions")
    analyze_predictions(multi, pred_noarg, tf_class)

    output_directory = "../../results/training_metrics/"
    output_filename_base = path.basename(dataset_filename)
    output_filename = output_filename_base + '_metrics_' + str(model_type) + '_nb_unit_' + str(
        nb_unit) + '_training_set_size_' + str(
        percent_training) + '_nb_epochs_' + str(nb_epochs) + '_batch_size_' + str(batch_size) + '_GPU_tensorflow_' + str(loss) + '_' + str(activation)
    if multi:
        output_filename = output_filename + '_multi.txt'
    else:
        output_filename = output_filename + '.txt'

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
   # if np.isnan(results[0]):
   #     print("LOSS IS NAN! LOOP AGAIN.")
    #    main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training, activation, loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VariabilityMiningClassifier V0.1')

    parser.add_argument('--dataset', default='BPIC15.csv', type=str, dest="dataset_filename",
                        help="Name of the dataset (default: BPIC15.csv)")
    parser.add_argument('--model_type', default="RNN", dest="model_type",
                        help="Choose amongst: RNN, LSTM or GRU (default: RNN")
    parser.add_argument('--multi', default=False, dest="multi",
                        help="Enable option to choose multi-labelling instead of single-labelling", action="store_true")
    parser.add_argument('--nb_epochs', default="50", dest="nb_epochs", type=int, help="Number of epochs (default=50)")
    parser.add_argument('--nb_units', default="10", dest="nb_unit", type=int,
                        help="Number of units for the model (default=10)")
    parser.add_argument('--training_ratio', default="0.66", dest="percent_training", type=float,
                        help="Ratio of the dataset used for training (default=0.66)")
    parser.add_argument('--batch_size', default="128", dest="batch_size",
                        help="Size of the batch for training (default=128)", type=int)
    parser.add_argument('--activation', default="tanh", dest="activation",
                        help="Activation function to use (default=tanh)", type=str)
    parser.add_argument('--loss', default="mse", dest="loss",
                        help="Loss function to use (default=mse)", type=str)

    args = parser.parse_args()

    dataset_filename = args.dataset_filename
    percent_training = args.percent_training
    nb_epochs = args.nb_epochs
    nb_unit = args.nb_unit
    batch_size = args.batch_size
    model_type = args.model_type
    multi = args.multi
    activation = args.activation
    loss = args.loss

    main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training, activation, loss)
