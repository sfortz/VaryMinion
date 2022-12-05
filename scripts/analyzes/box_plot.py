import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_match(match, regex):
    if match:
        x = match.group()
        l0 = re.sub(r'^' + re.escape(regex), '', x)
        s = re.sub(re.escape('_') + r'$', '', l0)
    else:
        print("BUG: " + str(match) + regex)
        s = ''
    return s


def parse_file_name(file_name):
    # print("Name : " + file_name)
    x = re.search(".*.csv", file_name).group()
    dataset = re.sub(re.escape('.csv') + r'$', '', x)

    x = re.search("metrics_[A-Z]*_", file_name)
    model = parse_match(x, 'metrics_')

    x = re.search("nb_epochs_[0-9]*_", file_name)
    nb_epoch = parse_match(x, 'nb_epochs_')

    x = re.search("nb_unit_[0-9]*_", file_name)
    nb_unit = parse_match(x, 'nb_unit_')

    x = re.search("batch_size_[0-9]*_", file_name)
    batch_size = parse_match(x, 'batch_size_')

    x = re.search("training_set_size_[0-9]*.[0-9]*_", file_name)
    training_set_size = parse_match(x, 'training_set_size_')

    x = re.search("_CPU|_GPU", file_name)
    proc = parse_match(x, '_')

    x = re.search("_tensorflow|_plaid_ml", file_name)
    backend = parse_match(x, '_')

    x = re.search("_[a-zA-Z]*$", file_name)
    layer = parse_match(x, '_')

    x = re.search(backend + "_.*_" + layer, file_name)
    if re.search("_sigmoid_", x.group()):
        x = parse_match(x, backend + '_')
        loss = re.sub("_sigmoid_" + layer, '', x)
        activation = "sigmoid"
    else:
        x = parse_match(x, backend + '_')
        loss = re.sub("_tanh_" + layer, '', x)
        activation = "tanh"

    return dataset, model, nb_epoch, nb_unit, batch_size, training_set_size, proc, backend, layer, loss, activation


def read_csv_file(metric_directory):
    input_directory = "../../results/csv_metrics/"
    directory = input_directory + metric_directory

    with os.scandir(directory) as it:
        data = []
        for txt_filename in it:
            if txt_filename.is_file() and txt_filename.name.endswith('.csv'):
                name = txt_filename.name[:-4]
                input = pd.read_csv(directory + txt_filename.name, skiprows=1, header=None)
                dataset, model, nb_epoch, nb_unit, batch_size, training_set_size, processor, backend, layer, loss, activation = parse_file_name(
                    name)
                data.append([dataset, model, loss, activation, input])
                frame = pd.DataFrame(data, columns=['Dataset', 'Model', 'Loss', 'Activation', 'Data'])
                frame.sort_values(by=['Dataset', 'Model', 'Loss', 'Activation'], inplace=True)
            else:
                print('WARNING:' + str(txt_filename) + 'is not a csv file and will be ignored.')
        return frame


def weighted_mean(distribution):

    support_directory = "training_support/"
    data = read_csv_file(support_directory)

    avgs = []
    for config in data.index:
        w = data['Data'][config].values
        d = distribution['Data'][config].values
        avg = []
        for weights, distrib in zip(w, d):
            weighted_avg = round(np.average(distrib, weights=weights), 4)
            avg.append(weighted_avg)
        avgs.append(avg)

    distribution['Data'] = pd.DataFrame(np.array(avgs))


def create_boxplot(directory, ds, metric):
    data = []
    labels = []
    dataset = ""

    for config in ds.index:
        dataset = ds['Dataset'][config]
        model = ds['Model'][config]
        loss = ds['Loss'][config]
        activation = ds['Activation'][config]
        labels += [str(model) + " " + str(loss) + " " + str(activation)]

        if metric == "Accuracy":
            data.append(ds['Data'][config][1])
        else:
            data.append(ds['Data'][config])

    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(data, labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Saving the figure.
    title = str(metric) + " for " + str(dataset)
    plt.title(title)

    plt.savefig("../../results/box_plots/" + directory + dataset + ".png", bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':

    loss_acc_time_directory = "training_loss_acc_time/"
    precision_directory = "training_precision/"
    recall_directory = "training_recall/"
    f1score_directory = "training_f1/"

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    directories = [loss_acc_time_directory, precision_directory, recall_directory, f1score_directory]

    for metric, dir in zip(metrics, directories):

        print("Generating boxplot for " + metric)
        data = read_csv_file(dir)
        if metric != "Accuracy":
            weighted_mean(data)

        bpic15, bpic20, claroline_dis_10, claroline_dis_50, claroline_rand_10, claroline_rand_50 = [x for _, x in
                                                                                                    data.groupby(data[
                                                                                                                     'Dataset'])]
        datasets = [bpic15, bpic20, claroline_dis_10, claroline_dis_50, claroline_rand_10, claroline_rand_50]

        for ds in datasets:
            create_boxplot(dir, ds, metric)
