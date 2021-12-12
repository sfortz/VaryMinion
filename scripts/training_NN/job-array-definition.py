import os
import training_Model as minion

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
datasets = ["claroline-dis_10.csv"]
models = ["GRU"]
epochs = [20]
batch_sizes = [128]
units = [10]
activations = ["tanh"]
losses = ["bin_ce", "bin_ce-logits", "mse", "jaccard", "manhattan"]
nb_iterations = 3

"""
datasets = ["BPIC15.csv", "BPIC20.csv", "claroline-dis_10.csv", "claroline-dis_50.csv"]
models = ["GRU", "LSTM"]
epochs = [10, 20, 30, 50]
batch_sizes = [64, 128, 256, 1024, 2048, 8192]
units = [10, 30, 50]
activations = ["tanh", "sigmoid"]
losses = ["bin_ce", "bin_ce-logits", "mse", "jaccard", "manhattan"]
nb_iterations = 10"""


def get_number_of_executions():
    size = len(datasets) * len(models) * len(epochs) * len(batch_sizes) * len(units) * len(activations) * len(losses)
    print("Number of possible configurations: " + str(size))


def print_available_params_values(param, param_name):
    values = ""
    for i in range(0, len(param) - 1):
        values += str(param[i]) + "; "
    values += str(param[len(param) - 1])
    print("The " + param_name + " parameter allows the following values: " + values)


def available_configs():
    i = 0
    configs = []
    for ds in datasets:
        for m in models:
            for e in epochs:
                for bs in batch_sizes:
                    for u in units:
                        for act in activations:
                            for l in losses:
                                # config = [i, ds, m, e, u, bs, act, l]
                                configs += [[i, ds, m, e, u, bs, act, l]]
                                i += 1
    return configs


def get_params():
    tab = str(idx)[::-1]
    ds = datasets[int(tab[0]) - 1]
    m = models[int(tab[1]) - 1]
    e = 20  # epochs[int(tab[2])-1]
    u = 10  # units[int(tab[3])-1]
    bs = 128  # batch_sizes[int(tab[4])-1]
    act = "tanh"  # activations[int(tab[5])-1]
    l = losses[int(tab[2]) - 1]  # losses[int(tab[6])-1]
    return ds, m, e, u, bs, act, l


if __name__ == "__main__":

    ds, m, e, u, bs, act, l = get_params()
    config_name = ds + ' with ' + m + ' model, ' + str(e) + ' epochs, ' + str(u) + ' units, ' + str(
        bs) + ' batch_size, ' + act + \
                  ' as activation function and ' + l + ' as loss function.'
    print("Begining exécution of the " + str(idx) + " th configuration: " + str(config_name))
    for i in range(0, nb_iterations):
        print("Exécution " + str(i) + " : ")
        minion.main(ds, m, True, e, u, bs, 0.66, act, l)
