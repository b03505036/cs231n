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

