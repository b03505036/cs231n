問題總集
-------
#Assignment 1
##KNN
### Problem input X = (N , Data), Y = (N,) , N for what? 
N for number of samples

### Problem dists should be what dim? 
(N of test,N of data)

### Problem how to make clone for one raw to fiill the matrix? 
np.tile(array,(raws,copy number))

### Again above , how to use broadcasting? 
Look the tutorial note on cs231n. [Broadcasting](http://cs231n.github.io/python-numpy-tutorial/#numpy-arrays)

### Problem how to sum only one direction? 
axis! axis=1,raw方向相加

### Problem advancing using broadcast skill for no loop matrix addition. 
broadcasting is avaliable only in multiplication and add.
Have to have "1" or "same size of one direction"
it's good to use reshape(1,-1),we shouldn't know the vector of the side.

### Problem ((X-Y)^2)^(1/2)  
-> stuck in the size of input -> should be example 500X"3000" thank for [CSDN](https://blog.csdn.net/qq_22812319/article/details/79677525)

### Problem find the most K common element in array 
[Stackoverflow](https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector)

### function argsort -> return the indice of sorted array 
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html?highlight=argsor#numpy.argsort) 
註：由小到大的indice，也就是說argsort完後取[:k]是k個最小的數字的indice

### function argmax -> return the indice of max value
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html?highlight=argmax#numpy.argmax) 

### function bincount-> return the times the element occurrences. indice is the element.
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html)

## Cross-validation

### function numpy.array_split is good for spliting array, even when it occur the number dividable
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html)

### Problem how to abstract one of the validation array and use the rest to train
[CSND的cross validation part](https://blog.csdn.net/qq_22812319/article/details/79677525)
[np.vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html?highlight=vstack#numpy.vstack)
[np.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html?highlight=hstack#numpy.hstack)

### tips: when using np.sum, We can use np.sum(more useful than sum) to add boolean to int

example: 

np.sum(5==5) -> 1

np.sum((1,2,3)==(2,2,3)) -> 0

np.sum([(1,2,3)]==[(2,2,3)]) ->2 
the shape can not be a vector!!

## linear_SVM

### bias trick
third: append the bias dimension of ones (i.e. bias trick) so that our SVM only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

### SVM, we have multiple classes . "each test" data we will give them "C" scores to choose the highest one.

### Problem How to calculate SVM's gredient 

[CSND 講解的非常棒](https://blog.csdn.net/CV_YOU/article/details/78077329)
[analytic solution 講解的非常棒](https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html)

only modify W when 1(xiWj - xiWyi +1>0) occur,which means sorting into wrong class


∇Wyi Li = - xiT(∑j≠yi 1(xiWj - xiWyi +1>0)) + 2λWyi （j = y[i]）(當w在yi那行的更新)


∇Wj Li = xiT 1(xiWj - xiWyi +1>0) + 2λWj , (j≠yi) (j != y[i]) (當w不在yi那行的更新)

Q: why the sigama is missing？ no missing look to [analytic solution](https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html)

### 