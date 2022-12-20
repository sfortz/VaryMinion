from csv_reader import read_csv_file
import matplotlib.pyplot as plt

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
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(data, labels=labels)
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Saving the figure.
    title = str(metric) + " for " + str(dataset)
    plt.title(title)

    plt.savefig("../../results/box_plots/" + directory + "figure_" + metric + "-" + dataset + ".png",
                bbox_inches="tight")
    plt.close(fig)


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

        bpic15, bpic20, claroline_dis_10, claroline_dis_50, claroline_rand_10, claroline_rand_50 = [x for _, x in
                                                                                                    data.groupby(data[
                                                                                                                     'Dataset'])]
        datasets = [bpic15, bpic20, claroline_dis_10, claroline_dis_50, claroline_rand_10, claroline_rand_50]

        for ds in datasets:
            create_boxplot(dir, ds, metric)
