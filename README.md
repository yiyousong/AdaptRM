## AdaptRM 

1.	**Description**  
AdaptRM has a multi-task framework for the synergetic learning of epitranscriptomes. It was enabled by the adaptive pooling and several standard convolution blocks. Trained for an integrated multi-task formulated by three case studies, the model can operate on both low-resolution and high-resolution datasets without further preprocessing the input primary sequence, and conduct tissue-specific and type-specific modification site prediction according to users’ needs. 

2.	**Requirements**  
Before prediction, please make sure the following packages are installed in the Python environment:  
python = 3.10.0  
torch = 1.12.0  
numpy = 1.22.3  
pandas = 1.4.4  
argparse = 1.4.0  
scikit-learn = 1.1.1  
autograd = 1.5.0  
listmodel = 0.2.1  

3.	**Input and output**  
* A test.py is provided for .fasta file inputs and returns a .csv prediction file.  
* The model takes input of shape [1, -1, 4] representing a query sequence and returns a vector of values ranged from 0 to 1 representing the score of its corresponding task.  
* The AdaptRM.py can train a AdaptRM model and three competing methods described in the paper from scratch on new datasets
* The human_tissue.py, transfer_learn.py, and zebrafish.py in the 'outdated' folder are outdated files used to train the models. These files are uploaded just as reference examples only

## References  
* Rush A., The Annotated Transformer. In Proceedings of Workshop for NLP Open Source Software (NLP-OSS), 2018. (https://doi.org/10.18653/v1/W18-2509)

* Dosovitskiy A., Beyer L., Kolesnikov A., et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations. 2021. (https://doi.org/10.48550/arXiv.2010.11929)

* Huang D., Song B., Wei J., et al., Weakly supervised learning of RNA modifications from low-resolution epitranscriptome data. Bioinformatics, 2021. 37: p. i222-i230. (https://doi.org/10.1093/bioinformatics/btab278)

* Trockman A. and Zico Kolter J. Patches Are All You Need? 2022. arXiv:2201.09792. (https://doi.org/10.48550/arXiv.2201.09792)



