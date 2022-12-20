import os
import pandas as pd
from datetime import datetime, timedelta
from csv_reader import parse_file_name


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
