import training_Model as minion

if __name__ == "__main__":

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC20.csv", "GRU", True, 20, 10, 128, 0.66, "tanh", "mse")

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC20.csv", "LSTM", True, 20, 10, 128, 0.66)
