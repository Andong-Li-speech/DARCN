### [A Recursive Network with Dynamic Attention for Monaural Speech Enhancement](https://arxiv.org/abs/2003.12973)

This project is the implementation of DARCN, which has been accepted by INTERSPEECH2020.
The network combines the psychological characteristic of human in dynamic
attention and recursive learning. Experimental results indicate that the proposed
method obtains consistent metric improvements than previous approaches.

### Usage
#### Step1: 
Put the noisy-clean pairs into ./Dataset/train and ./Dataset/dev

#### Step2: 
change the parameter settings according to your directory (within config.py)

#### Step3: 
Run json_extract.py to generate json files, which records the utterance file names for both training and validation set
```shell
# Run json_extract.py
json_extract.py
```
#### Step4:
Network Training
```shell
# Run main.py to begin network training 
main.py
```

#### Step5: 
Test
After training, the model with smallest MSE will be saved into ./Best_model, put the test mix utterance into ./Test/mix
```shell
# Run Test.py to test the model
Test.py
```

### Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

    @inproceedings{Li2020,
    author={Andong Li and Chengshi Zheng and Cunhang Fan and Renhua Peng and Xiaodong Li},
    title={{A Recursive Network with Dynamic Attention for Monaural Speech Enhancement}},
    year=2020,
    booktitle={Proc. Interspeech 2020},
    pages={2422--2426},
    doi={10.21437/Interspeech.2020-1513},
    url={http://dx.doi.org/10.21437/Interspeech.2020-1513}
    }
