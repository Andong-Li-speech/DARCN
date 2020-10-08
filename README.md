### [A Recursive Network with Dynamic Attention for Monaural Speech Enhancement](https://arxiv.org/abs/2003.12973)

This project is the implementation of DARCN, which has been accepted by INTERSPEECH2020.
The network combines the psychological characteristic of human in dynamic
attention and recursive learning. Experimental results indicate that the proposed
method obtains consistent metric improvements than previous approaches.

### Usage
#### Step1: 
Put the noisy-clean pairs into ./Dataset/train and ./Dataset/dev

#### Step2: 
change the parameter settings accroding to your directory (within config.py)

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

    @article{li2020recursive,
    title={A Recursive Network with Dynamic Attention for Monaural Speech Enhancement},
    author={Li, Andong and Zheng, Chengshi and Fan, Cunhang and Peng, Renhua and Li, Xiaodong},
    booktitle={INTERSPEECH},
    year={2020}
    }