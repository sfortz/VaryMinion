import training_Model as minion

if __name__ == "__main__":

    for i in range(0, 4):
         print("Exécution " + str(i) + " : ")
         minion.main("claroline-dis_10.csv", "GRU", True, 20, 10, 128, 0.66, "tanh", "bin_ce")

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