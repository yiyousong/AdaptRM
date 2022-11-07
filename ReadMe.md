1.	Description

AdaptRM has a multi-tasking framework for the synergetic learning of epitranscriptomes. It was enabled by the adaptive pooling and several standard convolution blocks. Trained for an integrated multi-task formulated by three case studies, the model can operate on both low-resolution and high-resolution datasets without further preprocessing the input primary sequence, and conduct tissue-specific and type-specific prediction according to users’ needs. 

2.	Requirements

Before prediction, please make sure the following packages are installed in the Python environment:
Python = 3.10.0
Pytorch = 1.12.0
numpy = 1.22.3
pandas = 1.4.4
argparse = 1.4.0
scikit-learn = 1.1.1
autograd = 1.5.0
listmodel = 0.2.1

3.	Input and output 
* The input object should be a list of [1, -1, 4] representing a query sequence.
* The output would be a vector of values. Each element ranges from 0 to 1 and represents the probability of its corresponding task.

a test.py is provided for transforming .fasta file into the above shaped input and returns a .csv prediction file
