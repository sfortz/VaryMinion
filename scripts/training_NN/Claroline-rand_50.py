import training_Model as minion

if __name__ == "__main__":

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("claroline-rand_50.csv", "GRU", True, 20, 30, 128, 0.66, "tanh", "jaccard")

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("claroline-rand_50.csv", "LSTM", True, 20, 30, 128, 0.66, "tanh", "jaccard")
