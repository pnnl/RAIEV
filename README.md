# RAIEv

RAIEV, the **R**esponsible **AI** **Ev**aluation package contains protypes of AI-assisted evaluation workflows and interactive analytics for model evaluation and understanding. 

</br>Searchable documentation is available at <a href="https://pnnl.github.io/RAIEV/" target="_blank"> https://pnnl.github.io/RAIEV/ </a>

</br>
</br>

<hr>


## Installation

- It is (highly) recommended to create a python 3 virtual environment for installation of the RAIEv package. The following instructions have been tested on python version 3.8. 
- There are two ways to accomplish this: one using conda (miniconda) and the other using python's venv package.

### Install RAIEv Package in Conda Environment

```sh

# Enter RAIEv source dir
cd package/

# Create conda env
conda env create --file environment.yml -n raiev

# Activate raiev conda env
conda activate raiev

# Install RAIEv package (via setup.py)
# NOTE: the --editable flag will allow you to edit code and regenerate docs without having to reinstall the package. 
pip install --editable .
```

### (Alternative to Conda) Python Venv-based Installation

- Prerequisites: pandoc, python3-pip and python3-venv packages are installed on the development machine or VM

```sh
# Enter RAIEv source dir
cd package/

# Create python 3 virtual environment under .venv directory
python3 -m venv .venv
# NOTE: if you have multiple versions of python 3 installed on your machine, you can select a specific version to use as follows:
python3.8 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify your python version is compatible (currently tested as 3.8.10)
python --version
# expected: Python 3.8.10

# Upgrade pip, currently 23.2
pip install --upgrade pip

# Install RAIEv package as editable (via setup.py)
# NOTE: the --editable flag will allow you to edit code and regenerate docs without having to reinstall the package. 
pip install --editable .
```

### Install Jupyter and Dev Environment

- After `raiev` is installed, you can install Jupyter and the remaining dependencies:

```sh
pip install -r requirements/requirements-default.txt
# or, for additional dev tools (like test coverage)
pip install -r requirements/requirements-all.txt
```

### Java tool for causal informed workflow

A Java-based command line tool is used for performing causal discovery in the causal informed workflow. Before running that workflow:
- Install Java. This tool was tested with the following version of Java:
    > openjdk 17.0.11 2024-04-16

    > OpenJDK Runtime Environment (build 17.0.11+9-Ubuntu-120.04.2)
    
    > OpenJDK 64-Bit Server VM (build 17.0.11+9-Ubuntu-120.04.2, mixed mode, sharing)
- Download the causal discovery tool jar file:
    >  https://s01.oss.sonatype.org/content/repositories/releases/io/github/cmu-phil/causal-cmd/1.12.0/causal-cmd-1.12.0-jar-with-dependencies.jar
 
### (Optional) Building Documentation + API Reference

- Note that you will need the `raiev` package installed as above, preferably as `--editable`.

- The following packages are needed to build the docs. From your virtual environment (conda or venv, see above):

```sh
conda activate raiev 

# Pandoc needs to be installed via conda (or by package manager if using python venv.)
conda install pandoc
pip install -r requirements/requirements-dev.txt
```

- You can then build the API docs with `make docs`. (You can ignore the various warnings for now.)

- The output can then be browsed from `docs/_build/html/index.html`
 

## Project Dependencies

 Dependencies are compiled and their versions pinned with [pip-tools](https://github.com/jazzband/pip-tools).
 To add or change a dependency, edit `requirements.in` (or `requirements-dev.in`, etc.), then run `make requirements`
 to update the automatically-generated `requirements.txt` (or `requirements-dev.txt`, etc.).
 


<hr>


This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830

