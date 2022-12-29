# VaryMinions

This repository gathers all materials that were used to conduct experiments to predict to which variant an execution trace belongs to. We experimented with both business processes and a Software Product Line (SPL). This repository contains the material for an EMSE extension of **VaryMinions**, which was first published at [MaLTeSQuE](https://maltesque2021.github.io/index.html).

## Prerequisites

Our experiments were ran on on two different HPC facilities hosted by the [CECI](http://www.ceci-hpc.be/). 
All our scripts are written in Python3 to ensure portability. We have used [Keras](https://keras.io/) along with the 
[TensorFlow](https://www.tensorflow.org/) framework.

## Running setup

On the first cluster, called Dragon1, we used 1  CPU  with  8  cores  per  task  (Intel  Sandy  Bridge,  E5-2650  processors  at2.00GHz) with a Tesla Kepler accelerator (K20m, 1.1 Tflops, float64). 
For runs on Dragon2, we used 1 CPU with 12 cores (Intel SkyLake, Xeon 6126 processors at 2.60 GHz) associated with a NVidia Tesla Volta V100 accelerator (5120 CUDA Cores, 16GB HBM2, 7.5 TFlops, double precision). 
On the third cluster, Hercules, we had access to 1 CPU with 8 cores per task (Intel Sandy Bridge,  Xeon E5-2660 processors at 2.20 GHz) with an NVidia GeForce accelerator (RTX 2080 Ti, 7.5 TFlops, double precision).
Each CPU has been allocated 3 GB of RAM. [Slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html) is the resource manager and job scheduler used on these infrastructures. To use the job scheduler, we can use the scripts _run-*.sh_ (in the directory root).

## Organisation of the repo

Hereafter, we list what can be found in this repository. We have decomposed its content into 3 different folders that we present in details.

### Datasets
This folder gathers the different Datasets that were used to conduct our experiments. As presented in the paper, we have used 6 different ones coming from the BPI Challenges (Resp. from 2015 and 2020 editions) and derived from the Claroline SPL.
If further datasets should be added (for reproducibility or apply our scripts on different datasets), they should be added here.

### Scripts

This folder contains all the scripts that were used to conduct experiments. It is composed of two different folders: _analyzes_ to study the characteristics of the different datasets and _training\_NN_ to train a specific RNN model and evaluate it with specific parameterization.

The _analyzes_ folder is composed of several files: 
 - _alphabet\_coverage_ is about knowing if the number of traces per process variant is balanced (which would ease further training) and if not, is there any risk that some process variant are not represented in the training set? To do so, we create a dictionary of symbols, prepare training and test sets as we were about to train a model and count the number of symbols that were not encountered in the training set. 
 - _DS\_analyse_ is a script conducting some analyzes to evaluate the difficulty to learn a classifier.
 - _writer\_latex_ is about generating tables (both in tex and pdf format) and creating plots out from analyzes.
 - _writer\_csv_ is simply a parser that turns output text files into csv.
 - _csv\_reader_ contains utilities to read csv metrics and ease analysis.
 - _box\_plot_ creates box-plots from the csv files.
 - _statistics_ performs a Friedman's test along with a Nemenyi's post-hoc analysis to compare the different parameterisations.
 - _stat.R_ is called by _box\_plot_ and _statistics_ to delegate part of the treatment to R.
 - _get\_total\_time_ retrieves the total time required to perform all executions.
 
The _Training_NN_ folder contains all scripts to learn different models with different parameterization on the datasets used in the paper. 
 - _training\_Model_ is some kind of interface that ease the training of any RNN following the same procedure. It is also in charge of managing the parameters given when the script is launched. This script is called by _DS*-*unit.py_ scripts.
 - _training\_GRU_ and _training\_LSTM_ are called by the _training\_Model_, depending on type of RNN selected.
 - _preprocessing_ is called by the _training\_Model_ to execute the conversion of traces into tensors.
 - _vary_minion_losses_ defines two custom loss functions: Manhattan and Jaccard distances.
 - _training\_GRU_ and _training\_LSTM_, _training\_RNN_ are called by the _training\_Model_.
 - _DS*-*unit.py_ and _Claroline-*.py_ scripts are examples of usage of the learning framework.  
 - _job-array-*.py_ scripts are examples of usage of the learning framework as an array job for SLURM.  
 
### Slurm-output

This folder contains all the output files from the clusters. Each file register the standard output of a specific job. 
They are very useful for and debugging purposes.

### Results

The _Results_ folder contains output files from our experiments:
- _csv\_metrics_ and _training\_metrics_ contain the same files (answering RQ1 and RQ2) except that the output format is either txt or csv. It stores accuracy, precision, recall, f1-score and support computed during training. _training\_metrics_ also stores confusion matrices which were not further analysed.
- _latex\_analyses_ contains tables with average and standard deviation for each of the metrics (accuracy, precision, recall, f1-score), register as pdf and LaTeX.   
- _box\_plots_ contains one figure per metric (accuracy, precision, recall, f1-score). In each figure, there is one box-plot per dataset.   
- _statistics_ contains a figure with the result of a multi-comparison statistical test: a Friedman's test with a Nemenyi's post-hoc explanation.

## Execution

There are three main ways to run the framework:
 - Using the _training\_Model_ entry point either from another script (see _DS*-*unit.py_ for examples) or 
   from the terminal: `python3 training_Model.py  --dataset BPIC15.csv --model_type LSTM --nb_epochs 10 --nb_units 10 --training_ratio 66‘.  This commands trains a LSTM with 10 epochs and 10 units and 66% of training data for the BPIC15.csv datasets.
 - Using _job-array-*.py_ scripts with SLURM job scheduler. Examples of the SLURM bash scripts we used are in the root of this repository (_run-dragon*.sh_).


