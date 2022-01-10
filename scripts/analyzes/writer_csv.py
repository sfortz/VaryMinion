import csv
import os
import re


def get_data(line):
    l0 = re.sub(r'^' + re.escape('test loss, test acc, exec time: ['), '', line)
    l1 = re.sub(re.escape(']') + r'$', '', l0)
    data = re.sub(re.escape('\''), '', l1)
    return data


def parse(txt_file, csv_file):
    in_file = open(txt_file, mode='r')
    out_file = open(csv_file, mode='w')

    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['test loss', 'test acc', 'exec time'])

    for line in in_file:
        row = get_data(line).split(',')
        writer.writerow(row)

    in_file.close()
    out_file.close()


if __name__ == "__main__":

    input_directory = "../../results/training_metrics/"
    output_directory = "../../results/csv_metrics/"

    with os.scandir(input_directory) as it:
        for txt_filename in it:
            print(txt_filename)
            if txt_filename.is_dir():
                print('Skipping directory.')
            elif txt_filename.is_file():
                if txt_filename.name.endswith('.txt'):  # and txt_filename.name.startswith('BPIC15'):
                    name = txt_filename.name[:-4]
                    csv_filename = name + '.csv'
                    txt_file = input_directory + txt_filename.name
                    csv_file = output_directory + csv_filename
                    parse(txt_file, csv_file)
            else:
                print('ERROR: Not a text file.')
