import training_Model as minion

if __name__ == "__main__":

    for i in range(0, 2):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC15.csv", "GRU", True, 20, 30, 128, 0.66, "tanh", "bin_ce")

    for i in range(0, 2):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC15.csv", "GRU", True, 20, 30, 128, 0.66, "tanh", "bin_ce-logits")

