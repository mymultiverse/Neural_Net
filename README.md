# Neural_Net Overview
Implementation of Artificial Neural Network, backpropagation from scratch with changeable number of layers and nodes. 

# Dependencies
* Python
* numpy
* matplotlib
* sklearn

Use [pip](https://pypi.org/project/pip/) to install any missing dependencies

# Uses
Make sure input data is in proper formate dimentions (no. of example X features). Default run uses IRIS data set.  
Clone or download repository. Inside repo. run
```markdown
python test.py
```
In case of binary classification use Y without one-hot encoding function inside class file.

# Results
[Iris data](https://archive.ics.uci.edu/ml/datasets/Iris)

![](https://github.com/mymultiverse/Neural_Net/blob/master/sig.png)
```markdown
Training accuracy:100.0%
Test Accuracy:94.0%
```
[Breast Cancer data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

![](https://github.com/mymultiverse/Neural_Net/blob/master/tloss.png)
```markdown
    Training accuracy:97.36%
    Test Accuracy:97.34%
```

Todo:

MNIST data

# Reference:-
[Courses](https://www.deeplearning.ai/)

[Backpropagation](http://cs231n.github.io/optimization-2/)
