import training_Model_tensorflow as minion

if __name__ == "__main__":

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC15.csv", "GRU", True, 20, 30, 128, 0.66)

    for i in range(0, 10):
        print("Exécution " + str(i) + " : ")
        minion.main("BPIC15.csv", "LSTM", True, 20, 30, 128, 0.66)
