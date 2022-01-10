import os
import training_Model as minion

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

if __name__ == "__main__":

    if idx == 1:
        for i in range(0, 2):
             print("Exécution " + str(i) + " : ")
             minion.main("claroline-dis_10.csv", "GRU", True, 20, 30, 128, 0.66, "tanh", "jaccard")

    if idx == 2:
        for i in range(0, 2):
             print("Exécution " + str(i) + " : ")
             minion.main("claroline-dis_10.csv", "LSTM", True, 20, 30, 128, 0.66, "tanh", "bin_ce")
   # for i in range(0, 10):
   #     print("Exécution " + str(i) + " : ")
    #    minion.main("claroline-dis_10.csv", "LSTM", True, 20, 30, 128, 0.66, "tanh", "manhattan")

   # for i in range(0, 10):
   #      print("Exécution " + str(i) + " : ")
   #      minion.main("claroline-dis_10.csv", "GRU", True, 20, 10, 128, 0.66, "tanh", "manhattan")

   # for i in range(0, 10):
   #     print("Exécution " + str(i) + " : ")
   #     minion.main("claroline-dis_10.csv", "LSTM", True, 20, 10, 128, 0.66, "tanh", "manhattan")

   # for i in range(0, 10):
   #     print("Exécution " + str(i) + " : ")
    #    minion.main("claroline-dis_10.csv", "GRU", True, 20, 5, 128, 0.66, "tanh", "manhattan")

   # for i in range(0, 10):
  #      print("Exécution " + str(i) + " : ")
   #     minion.main("claroline-dis_10.csv", "LSTM", True, 20, 5, 128, 0.66, "tanh", "manhattan")