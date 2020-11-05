import argparse
from subprocess import check_output


def main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training, keras_backend='tensorflow'):
    # Since tensorflow and PlaidML do not behave well in the same code without issues, we provide two different and
    # exclusive implementations for each backend. We need to run them as separate calls as they cannot depend on each
    # other.
    cmd = []
    if keras_backend == 'tensorflow':
        cmd = ["python3", "training_Model_tensorflow.py", "--dataset", str(dataset_filename), "--model_type",
               str(model_type), "--multi", "--nb_epochs", str(nb_epochs), "--nb_units", str(nb_unit),
               "--batch_size",str(batch_size), "--training_ratio", str(percent_training)]
    elif keras_backend == 'plaidml':
        cmd = ["python3", "training_Model_plaidml.py", "--dataset", str(dataset_filename), "--model_type",
               str(model_type), "--multi", "--nb_epochs", str(nb_epochs), "--nb_units", str(nb_unit),
               "--batch_size",str(batch_size), "--training_ratio", str(percent_training)]


    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(cmd)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    c = check_output(cmd)
    # for debug only...
    if c is not None:
        print(c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VariabilityMiningClassifier V0.1')

    parser.add_argument('--dataset', default='BPIC15.csv', type=str, dest="dataset_filename",
                        help="Name of the dataset (default: BPIC15.csv)")
    parser.add_argument('--model_type', default="RNN", dest="model_type",
                        help="Choose amongst: RNN, LTSM or GRU (default: RNN")
    parser.add_argument('--multi', default=False, dest="multi",
                        help="Enable option to choose multi-labelling instead of single-labelling", action="store_true")
    parser.add_argument('--nb_epochs', default="50", dest="nb_epochs", type=int, help="Number of epochs (default=50)")
    parser.add_argument('--nb_units', default="10", dest="nb_unit", type=int,
                        help="Number of units for the model (default=10)")
    parser.add_argument('--training_ratio', default="0.66", dest="percent_training", type=float,
                        help="Ratio of the dataset used for training (default=0.66)")
    parser.add_argument('--batch_size', default="128", dest="batch_size",
                        help="Size of the batch for training (default=128)", type=int)
    parser.add_argument('--backend', default='tensorflow', dest='keras_backend', type=str,
                        help='Backend to be used for Keras. tensorflow (default), plaidml = plaidml backend (for AMD '
                             'GPUs)')
    args = parser.parse_args()

    dataset_filename = args.dataset_filename
    percent_training = args.percent_training
    nb_epochs = args.nb_epochs
    nb_unit = args.nb_unit
    batch_size = args.batch_size
    model_type = args.model_type
    keras_backend = args.keras_backend
    multi = args.multi

    main(dataset_filename, model_type, multi, nb_epochs, nb_unit, batch_size, percent_training, keras_backend)
