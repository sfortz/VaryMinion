import os
import re
import numpy as np
import pandas as pd
from pylatex import Document, Section, LongTable, NewPage
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

    x = re.search("_CPU|_GPU_.*X", file_name)
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
        loss = re.sub("_" + layer, '', x)
        activation = "tanh"

    return dataset, model, nb_epoch, nb_unit, batch_size, training_set_size, proc, backend, layer, loss, activation


def analyze(in_file):
    data = pd.read_csv(in_file)
    values = list(data['test acc'])
    mean = np.around(np.mean(values, dtype=np.float64), 5)
    deviation = np.around(np.std(values, dtype=np.float64), 5)

    return mean, deviation


def generate(bpic15, bpic20, hosp_bill, out_file):
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)

    with doc.create(Section('Table of results for BPIC15')):
        with doc.create(LongTable('|c|c|c|c|c|c|c|')) as table:  # , angle=270
            table.add_hline()
            table.add_row(bpic15.columns, mapper=bold, color="lightgray!70")
            table.add_hline()
            for index, row in bpic15.iterrows():
                table.add_row(row)
                table.add_hline()

    doc.append(NewPage())

    with doc.create(Section('Table of results for BPIC20')):
        with doc.create(LongTable('|c|c|c|c|c|c|c|')) as table:  # , angle=270
            table.add_hline()
            table.add_row(bpic20.columns, mapper=bold, color="lightgray!70")
            table.add_hline()
            for index, row in bpic20.iterrows():
                table.add_row(row)
                table.add_hline()

    doc.append(NewPage())

    with doc.create(Section('Table of results for Hospital Billing')):
        with doc.create(LongTable('|c|c|c|c|c|c|c|')) as table:  # , angle=270
            table.add_hline()
            table.add_row(hosp_bill.columns, mapper=bold, color="lightgray!70")
            table.add_hline()
            for index, row in hosp_bill.iterrows():
                table.add_row(row)
                table.add_hline()

    doc.generate_tex(out_file + "_tex")
    doc.generate_pdf(out_file + "_pdf")


if __name__ == '__main__':
    input_directory = "../../results/csv_metrics/"
    output_directory = "../../results/results_analyze/"

    data = []

    with os.scandir(input_directory) as it:
        # print('Name = ' + it.name)
        for txt_filename in it:
            if txt_filename.is_file() and txt_filename.name.endswith('.csv'):
                input = input_directory + txt_filename.name
                name = txt_filename.name[:-4]
                dataset, model, nb_epoch, nb_unit, batch_size, training_set_size, processor, backend, layer, loss, activation = parse_file_name(name)
                m, d = analyze(input)
                data.append([dataset, model, loss, activation, nb_unit, m, d])  # nb_epoch, batch_size, training_set_size, processor, backend, layer,

            else:
                print('ERROR:' + str(txt_filename) + 'is not a csv file.')

    frame = pd.DataFrame(data, columns=['Dataset', 'Model', 'Loss', 'Activation', 'Units', 'Mean', 'Sd'])
    # frame.sort_values(by=['Dataset', 'Model', 'Type', 'Epochs', 'Units'], inplace=True)
    frame.sort_values(by=['Mean'], inplace=True, ascending=False)
    bpic15, bpic20, hosp_bill = [x for _, x in frame.groupby(frame['Dataset'])]

    out_filename = 'results_analyse'
    output = output_directory + out_filename
    generate(bpic15, bpic20, hosp_bill, output)
