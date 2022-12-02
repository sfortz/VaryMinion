import csv
import os
import re


def parse_loss_acc_time(txt_file, csv_file):
    in_file = open(txt_file, mode='r')
    out_file = open(csv_file, mode='w')

    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['test loss', 'test acc', 'exec time'])

    for line in in_file:
        data = re.sub("test loss, test acc, exec time:\[", '', line, 1)
        data = re.sub("\]", '', data, 1)
        data = re.sub("\'", '', data)
        data = re.sub("\n", '', data)
        row = data.split(',')
        writer.writerow(row)

    in_file.close()
    out_file.close()


def parse_metric(txt_file, csv_file, metric):
    in_file = open(txt_file, mode='r')
    out_file = open(csv_file, mode='w')

    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([metric])

    txt = ""

    for line in in_file:
        txt = txt + line
        txt = re.sub(metric + ":\[", '', txt, 1)

    for row in txt.split(']'):
        if metric == "support":
            regex = r"\d+"
        else:
            regex = r"\d\.\d*|\d\.\d*e[+|-]\d\d"

        scores = re.findall(regex, row)
        writer.writerow(scores)

    in_file.close()
    out_file.close()


def converting(in_dir, out_dir, metric):
    with os.scandir(in_dir) as it:
        for txt_filename in it:
            # print(txt_filename)
            if txt_filename.is_dir():
                print('Skipping directory.')
            elif txt_filename.is_file():
                if txt_filename.name.endswith('.txt'):
                    name = txt_filename.name[:-4]
                    csv_filename = name + '.csv'
                    txt_file = in_dir + txt_filename.name
                    csv_file = out_dir + csv_filename

                    if metric == "loss_acc_time":
                        parse_loss_acc_time(txt_file, csv_file)
                    else:
                        parse_metric(txt_file, csv_file, metric)

            else:
                print('ERROR: Not a text file.')


if __name__ == "__main__":
    input_directory = "../../results/training_metrics/"
    output_directory = "../../results/csv_metrics/"

    loss_acc_time_directory = "training_loss_acc_time/"
    precision_directory = "training_precision/"
    recall_directory = "training_recall/"
    f1score_directory = "training_f1/"
    conf_matrix_directory = "training_conf_matrix/"
    support_directory = "training_support/"

    converting(input_directory + loss_acc_time_directory, output_directory + loss_acc_time_directory, "loss_acc_time")
    converting(input_directory + precision_directory, output_directory + precision_directory, "precision")
    converting(input_directory + recall_directory, output_directory + recall_directory, "recall")
    converting(input_directory + f1score_directory, output_directory + f1score_directory, "f1score")
    converting(input_directory + support_directory, output_directory + support_directory, "support")
    # converting(input_directory + conf_matrix_directory, output_directory + conf_matrix_directory, "conf_matrix")
