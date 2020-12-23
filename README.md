This repository contains the code for the paper "Understanding the Mechanics of SPIGOT: Surrogate Gradients for Latent Structure Learning" by Tsvetomila Mihaylova, Vlad Niculae and André F. T. Martins.

https://www.aclweb.org/anthology/2020.emnlp-main.171/


## Requirements

* The file **requirements.txt** contains required libraries.

* You also need to install [pytorch](https://pytorch.org/).

* We use [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) for the experiments with structured latent variables.

* We use SparseMAP in the experiments with structured latent variables. *Instructions how to install the necessary dependencies will be added soon.*


## Organization of the code

The code for the three groups of experiments in the paper are organized in separate folders:

* **synthetic** - contains code for the experiments with categorical latent variable with synthetic data.

* **sst** - contains the code for sentiment classification with latent syntax

* **nli** - congtains code for natural language inference with latent syntax


## Running the experiments

Each directory has a **main.py** file which needs to be executed. 

There are shell scripts with examples of the necessary parameters for the experiments with structured variables.

The parameters for the experiments with categorical latent variable are set in the Python file.

*More details will be added soon.*


## Questions

If you have any questions or problems running the code, please let me know. You can post an issue in the repository or email the first author on *firstname.lastname @ gmail . com*. 





