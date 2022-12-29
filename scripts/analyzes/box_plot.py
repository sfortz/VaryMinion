import pandas as pd

from csv_reader import read_csv_file
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def weighted_mean(distribution):
    support_directory = "training_support/"
    data = read_csv_file(support_directory)

    for config in data.index:
        w = data['Data'][config]
        d = distribution['Data'][config]
        weighted_distrib = round(d * w, 4)
        distribution['Data'][config] = weighted_distrib.sum(axis=1) / w.sum(axis=1)


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
    bp = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp.boxplot(data, labels=labels)
    plt.setp(bp.get_xticklabels(), rotation=90)

    # Saving the figure.
    title = str(metric) + " for " + str(dataset)
    plt.title(title)

    plt.savefig("../../results/box_plots/" + directory + "figure_" + metric + "-" + dataset + ".png",
                bbox_inches="tight")
    plt.close(fig)


def get_dataset(ds):

    colNames = ["learner", "dataset", "metric"]

    frames = []
    for config in ds.index:
        dataset = ds['Dataset'][config]
        model = ds['Model'][config]
        loss = ds['Loss'][config]
        activation = ds['Activation'][config]

        sep = " "
        rows = pd.DataFrame(index=range(0, 10), columns=colNames)
        rows["learner"] = model + sep + loss + sep + activation
        rows["dataset"] = dataset

        if(metric == "Accuracy"):
            rows["metric"] = ds['Data'][config][1]
        else:
            rows["metric"] = ds['Data'][config]

        frames.append(rows)

    df = pd.concat(frames, ignore_index=True)
    return df


def create_r_boxplot(ds, metric):

    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('stat.R')
    # Loading the function we have defined in R.
    r_function = ro.globalenv['boxplots']

    output_directory = "../../results/box_plots"

    frame = get_dataset(ds)
    importr('base')
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        #Invoking the R function
        r_function(frame, output_directory, metric)


if __name__ == '__main__':

    loss_acc_time_directory = "training_loss_acc_time/"
    precision_directory = "training_precision/"
    recall_directory = "training_recall/"
    f1score_directory = "training_f1/"

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    directories = [loss_acc_time_directory, precision_directory, recall_directory, f1score_directory]

    for metric, dir in zip(metrics, directories):

        print("Generating boxplot for " + metric)
        data = read_csv_file(dir)
        if metric != "Accuracy":
            weighted_mean(data)

        create_r_boxplot(data, metric)