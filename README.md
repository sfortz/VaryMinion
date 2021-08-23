# VaryMinions

This repository gathers all materials that were used to conduct experiments to predict to which variant an execution trace belongs to. We so far experimented with business processes.    

## Prerequisites

Our experiments were ran on 3 different machines which had similar materials as indicated in the paper (i.e., 16 GB RAM, Intel core i7 processor) but were using different Operating Systems (MacOS and Fedora).
All our scripts are written in Python3 to ensure portability.
We have used [Keras](https://keras.io/) along with the TensorFlow and [PlaidML](https://github.com/plaidml/plaidml) backends. PlaidML allows to run experiments non-Nvidia GPUs (i.e., Intel and AMD), which notably enables to run experiments on Apple laptops. 
To ease portability and avoid conflicts between backends, we provided two separate implementations, one for tensorflow and one for plaidML.  

## Organisation of the repo

Hereafter, we list what can be found in this repository. We have decomposed its content into 3 different folders that we present in details.

### Datasets
This folder gathers the different Datasets that were used to conduct our experiments. As presented in the paper, we have used 3 different ones coming from the BPI Challenge and a last one which is known as HospitalBilling.
If further datasets should be added (for reproducibility or apply our scripts on different datasets), they should be added here.

### Scripts

This folder contains all the scripts that were used to conduct experiments. It is composed of two different folders: _analyzes_ to study the characteristics of the different datasets and _training\_NN_ to train a specific RNN model and evaluate it with specific parameterization.

The _analyzes_ folder is composed of several files: 
 - _alphabet\_coverage_ is about knowing if the number of traces per process variant is balanced (which would ease further training) and if not, is there any risk that some process variant are not represented in the training set? To do so, we create a dictionary of symbols, prepare training and test sets as we were about to train a model and count the number of symbols that were not encountered in the training set. 
 - _DS\_analyse_ is the script answering RQ3 in the paper (conducting some analyzes to evaluate the difficulty to learn a classifier).
 - _results\_analyse_ is about generating tables (both in tex and pdf format) and creating plots out from analyzes.
 - _writer\_csv_ is simply a parser that turns output text files into csv.
 
The _Training_NN_ folder contain all scripts to learn different models with different parameterization on the datasets used in the paper. To distinguish between implementations, files are suffixed by the name of the backend to be used (_\_plaidML_ or _\_tensorflow_). Files with no suffix are backend-independent. 
 - _training\_Model_ is some kind of interface that ease the training of any RNN following the same procedure. It is also in charge of the managing the parameters given when the script is launched. This script is called by _DS*-*unit.py_ scripts.
 - _training\_GRU_ and _training\_LSTM_, _training\_RNN_ are called by the _training\_Model_, depending on type of RNN selected.
 - _DS*-*unit.py_ scripts are examples of usage of the learning framework.
 - Finally, _vary\_minion_ is the entry point to call both implementations easily.  
 
 ### Tests
 
 This folder contains a few tests (notably to ensure that plaidML is correctly working. On a Mac laptop you can monitor GPU usage with Activity Monitor -> Display -> GPU operations, to be sure your are using the GPU and not the CPU).
 
## Execution

There are two main ways to run the framework:
 - Using the _vary\_minion_ entry point either from another script (see _DS*-*unit.py_ for examples) or 
   from the terminal: `python3 vary_minion.py  --dataset BPIC15.csv --model_type LSTM --nb_epochs 10 --nb_units 10 --training_ratio 66 --backend plaidml'.  This commands trains a LSTM with 10 epochs and 10 units and 66% of training data for the BPIC15.csv datasets using the plaidML backend. 
 - Using  _training\_Model_ interfaces of the backend of your choice, same options as above (choice of backend excluded, of course).  

### Results

The _Results_ folder contains output files from our experiments. It follows a similar structure as the one that can be found in the _scripts_ folder:
- _DS\_analyze_ is about numbers exploited to answer RQ3. It contains metrics characterising class overlap as a proxy for single class learning difficulty.   
- _csv\_metrics_ and _training\_metrics_ contain the same files (answering RQ1 and RQ2) except that the output format is either txt or csv. It represents accuracy and loss function results of training. 
- _training\_archives_ contains older results and other results not exploited in the paper.
- _results\_analyze_ and _alphabet\_coverage_ contain output files corresponding to scripts that can be found in the respective corresponding folders under _scripts_.


