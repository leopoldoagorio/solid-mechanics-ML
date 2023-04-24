# mamaML
First tests in automating a pipeline for generating automatically a database of FEM solutions to unidimensional compression/extension problem, and learning with a Neural Network to solve same mechanical problem.

The loop that generates the data is in the file [`download_data.sh`](./uniaxial_compression/data/loop.sh) and the script that trains the Neural Network is in [`surrogateMLP.py`](./uniaxial_compression/ML_model/surrogateMLP.py). The data is stored in the folder [`data`](./uniaxial_compression/data).

The final documentation for the project is in the file [`documentacion.pdf`](./documentation/documentacion.pdf).