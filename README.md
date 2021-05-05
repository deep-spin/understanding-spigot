This repository contains the code for the paper "Understanding the Mechanics of SPIGOT: Surrogate Gradients for Latent Structure Learning" by Tsvetomila Mihaylova, Vlad Niculae and André F. T. Martins.

https://www.aclweb.org/anthology/2020.emnlp-main.171/


## Requirements

* The file **requirements.txt** contains required libraries. You can install them with:

```
pip install -r requirements.txt
```

* You also need to install [pytorch](https://pytorch.org/), version 1.4

* We use [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) for the experiments with structured latent variables. Version v0.4 is required. Download the source code for v0.4 from the tags, navigate to the directory and install with: 

  ```
  pip install -e .
  ```


* We use SparseMAP in the experiments with structured latent variables. Here's how to setup it:

  - Download [lp-sparsemap](https://github.com/deep-spin/lp-sparsemap)
  ```
  git clone https://github.com/deep-spin/lp-sparsemap
  cd lp-sparsemap
  ```

  - Install Cython
  ```
  pip install Cython
  ```

  - Download Eigen and point to it in your environment
  ```
  git clone https://gitlab.com/libeigen/eigen
  export EIGEN_DIR=path/to/eigen
  ```

  - Activate the environment you are using for your project (if you are using one).

  - In the lp-sparsemap folder execute the following commands:

  ```
  python setup.py build_clib
  pip install -e .
  ```

* Download spacy 

  ```
  python -m spacy download en_core_web_sm
  ```

  - The setup should work. If it doesn't, don't hesitate to ask us. 

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





