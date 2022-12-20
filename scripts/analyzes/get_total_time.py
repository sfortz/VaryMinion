import os
import re
import pandas as pd
from datetime import datetime, timedelta


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
    print("Name : " + file_name)
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


def get_time(data):

    zero_time = datetime.strptime('0:0:0', '%H:%M:%S')
    values = data['exec time'].apply(lambda x: datetime.strptime(x, " %H:%M:%S"))
    deltas = [v - zero_time for v in values]
    time = sum(deltas, timedelta())
    return time


if __name__ == '__main__':
    input_directory = "../../results/csv_metrics/training_loss_acc_time/"
    output_directory = "../../results/latex_analyses/"

    data = []

    total_time = 0
    time = timedelta()
    with os.scandir(input_directory) as it:
        for txt_filename in it:
            if txt_filename.is_file() and txt_filename.name.endswith('.csv'):
                input = pd.read_csv(input_directory + txt_filename.name)
                if len(input) >= 10:
                    name = txt_filename.name[:-4]
                    dataset, model, nb_epoch, nb_unit, batch_size, training_set_size, processor, backend, layer, loss, activation = parse_file_name(
                        name)
                    time += get_time(input)

            else:
                print('ERROR:' + str(txt_filename) + 'is not a csv file.')
        print("total time: ", time)
