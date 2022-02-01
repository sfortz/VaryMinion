import os
import training_Model as minion

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

if __name__ == "__main__":

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("claroline-rand_10.csv", "GRU", True, 20, 30, 128, 0.66, "tanh", "jaccard")

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("claroline-rand_10.csv", "LSTM", True, 20, 30, 128, 0.66, "tanh", "jaccard")
