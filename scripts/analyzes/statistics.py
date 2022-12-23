import pandas as pd
from csv_reader import read_csv_file

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def r_stat(df, m, n, pvalue, title):
    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('Nemenyi.R')
    # Loading the function we have defined in R.
    r_function = ro.globalenv['multiCompare']

    output_directory = "../../results/statistics"

    importr('base')

    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):

        # Invoking the R function and getting the result
        r_function(df, m, pvalue, output_directory, title)



def get_dataset(ds):
    col_names = ["learners", "tasks", "accuracy"]

    frames = []
    for config in ds.index:
        dataset = ds['Dataset'][config]
        model = ds['Model'][config]
        loss = ds['Loss'][config]
        activation = ds['Activation'][config]
        data = ds['Data'][config][1]
        sep = " "
        rows = pd.DataFrame(index=range(0, 10), columns=col_names)
        rows["learners"] = model + sep + loss + sep + activation

        tasks = []
        for j in range(1, 11):
            tasks += [dataset + '-ite_' + str(j)]

        rows["tasks"] = tasks
        rows["accuracy"] = round(data, 4)
        frames.append(rows)

    temp = pd.concat(frames, ignore_index=True)
    row_names = temp["tasks"].drop_duplicates().values
    col_names = temp["learners"].drop_duplicates().values
    df = pd.DataFrame(index=row_names, columns=col_names)

    for obs, config, acc in zip(temp["tasks"], temp["learners"], temp["accuracy"]):
        df[config][obs] = acc

    df = df.astype("float64")
    return df


def r_launcher():
    # Reading and processing data
    loss_acc_time = read_csv_file("training_loss_acc_time/")
    df = get_dataset(loss_acc_time)

    title = "All_Configs_Accuracy"

    r_stat(df, 'accuracy', 10, 0.05, title)


if __name__ == '__main__':
    r_launcher()
