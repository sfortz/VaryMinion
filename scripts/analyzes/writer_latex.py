import os
import re
import pandas as pd
from pylatex import Document, Section, NewPage, MultiColumn, Tabular
from pylatex.utils import bold


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
    #print("Name : " + file_name)
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


def weighted_mean_sd(ds, metric):

    df_mean = ds.copy()
    del df_mean['Data']
    df_mean[metric] = ''
    df_sd = df_mean.copy()

    support_directory = "training_support/"
    data = read_csv_file(support_directory)

    for config in data.index:
        w = data['Data'][config]
        d = ds['Data'][config]
        weighted_distrib = round(d * w, 4)
        mean_per_it = weighted_distrib.sum(axis=1) / w.sum(axis=1)
        df_mean[metric][config] = round(mean_per_it.mean(axis=0), 4)
        df_sd[metric][config] = round(mean_per_it.std(axis=0), 4)

    merged = pd.merge(df_mean, df_sd, on=["Dataset", "Model", "Loss", "Activation"], how="outer", suffixes=("_avg", "_sd"))
    return merged


def mean_sd(ds, metric):

    df_mean = ds.copy()
    del df_mean['Data']
    df_mean[metric] = ''
    df_sd = df_mean.copy()

    for config in ds.index:
        data = ds['Data'][config][1]
        df_mean[metric][config] = round(data.mean(axis=0), 4)
        df_sd[metric][config] = round(data.std(axis=0), 4)

    merged = pd.merge(df_mean, df_sd, on=["Dataset", "Model", "Loss", "Activation"], how="outer", suffixes=("_avg", "_sd"))
    return merged


def write_ds(ds, ds_name, doc):

    with doc.create(Section('Table of results for ' + ds_name)):
        with doc.create(Tabular('|c|c|c|c|c|c|c|c|c|c|c|c|')) as table:  # , angle=270
            table.add_hline()
            col_acc = MultiColumn(2, align='c|', data=bold('Accuracy'))
            col_pre = MultiColumn(2, align='c|', data=bold('Precision'))
            col_rec = MultiColumn(2, align='c|', data=bold('Recall'))
            col_f1 = MultiColumn(2, align='c|', data=bold('F1Score'))
            row_titles = (bold('Dataset'), bold('Model'), bold('Loss'), bold('Activation'), col_acc, col_pre, col_rec, col_f1)
            table.add_row(row_titles, color="lightgray!70")
            table.add_hline()
            row_titles = ('', '', '', '', 'Avg', 'Sd', 'Avg', 'Sd', 'Avg', 'Sd', 'Avg', 'Sd')
            table.add_row(row_titles, color="lightgray!70", mapper=bold)
            table.add_hline()

            sorted_ds = ds.sort_values(by='Accuracy_avg', ascending=False)
            for index, row in sorted_ds.iterrows():
                table.add_row(row)
                table.add_hline()


def generate(bpic15, bpic20, claroline_dis_10, claroline_rand_10, claroline_dis_50, claroline_rand_50, out_file):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"} # , 'landscape': ""}
    doc = Document(geometry_options=geometry_options)

    write_ds(bpic15, "BPIC15", doc)
    doc.append(NewPage())
    write_ds(bpic20, "BPIC20", doc)
    doc.append(NewPage())
    write_ds(claroline_dis_10, "Claroline Dissimilar 10", doc)
    doc.append(NewPage())
    write_ds(claroline_rand_10, "Claroline Random 10", doc)
    doc.append(NewPage())
    write_ds(claroline_dis_50, "Claroline Dissimilar 50", doc)
    doc.append(NewPage())
    write_ds(claroline_rand_50, "Claroline Random 50", doc)
    doc.append(NewPage())

    doc.generate_tex(out_file + "_tex")
    doc.generate_pdf(out_file + "_pdf")

if __name__ == '__main__':

    loss_acc_time = read_csv_file("training_loss_acc_time/")
    precision = read_csv_file("training_precision/")
    recall = read_csv_file("training_recall/")
    f1score = read_csv_file("training_f1/")

    acc_mean_sd = mean_sd(loss_acc_time, "Accuracy")
    precision_mean_sd = weighted_mean_sd(precision, "Precision")
    recall_mean_sd = weighted_mean_sd(recall, "Recall")
    f1score_mean_sd = weighted_mean_sd(f1score, "F1Score")

    acc_prec = pd.merge(acc_mean_sd, precision_mean_sd, on=["Dataset", "Model", "Loss", "Activation"], how="outer")
    recall_f1 = pd.merge(recall_mean_sd, f1score_mean_sd, on=["Dataset", "Model", "Loss", "Activation"], how="outer")

    merged = pd.merge(acc_prec, recall_f1, on=["Dataset", "Model", "Loss", "Activation"], how="outer")

    output_directory = "../../results/latex_analyses/"

    bpic15, bpic20, claroline_dis_10, claroline_dis_50, claroline_rand_10, claroline_rand_50 = [x for _, x in merged.groupby(merged['Dataset'])]
    out_filename = 'results_analyse'
    output = output_directory + out_filename
    generate(bpic15, bpic20, claroline_dis_10, claroline_rand_10, claroline_dis_50, claroline_rand_50, output)
