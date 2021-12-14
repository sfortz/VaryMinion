import os
import training_Model as minion

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
datasets = ["claroline-dis_10.csv"]
models = ["GRU", "LSTM"]
epochs = [20]
batch_sizes = [128]
units = [30]
activations = ["tanh", "sigmoid"]
losses = ["bin_ce", "bin_ce-logits", "mse", "jaccard", "manhattan"]
nb_iterations = 10

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


def idx_to_config():
    tmp = idx - 1
    prod = 1
    result = 0
    values = []

    for param in ([losses, activations, batch_sizes, units, epochs, models, datasets]):
        tmp -= result
        mod = len(param)
        result = (tmp // prod) % mod
        prod *= mod
        values.insert(0, param[result])

    return values[0], values[1], values[2], values[3], values[4], values[5], values[6]


def get_params():
    tab = str(idx)
    print("tab = " + tab)
    ds = datasets[int(tab[0]) - 1]
    m = models[int(tab[1]) - 1]
    e = epochs[int(tab[2])-1]
    u = units[int(tab[3])-1]
    bs = batch_sizes[int(tab[4])-1]
    act = activations[int(tab[5])-1]
    l = losses[int(tab[6])-1]
    return ds, m, e, u, bs, act, l


if __name__ == "__main__":

    get_number_of_executions()

    ds, m, e, u, bs, act, l = idx_to_config()
    config_name = ds + ' with ' + m + ' model, ' + str(e) + ' epochs, ' + str(u) + ' units, ' + str(
        bs) + ' batch_size, ' + act + ' as activation function and ' + l + ' as loss function.'
    print("Begining exécution of the " + str(idx) + " th configuration: " + str(config_name))
    for i in range(0, nb_iterations):
        print("Exécution " + str(i) + " : ")
        minion.main(ds, m, True, e, u, bs, 0.66, act, l)
