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

### function argmax -> return the indice of max value
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html?highlight=argmax#numpy.argmax) 

### function bincount-> return the times the element occurrences. indice is the element.
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html)

## Cross-validation

### function numpy.array_split is good for spliting array, even when it occur the number dividable
[Numpy 文檔](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html)

