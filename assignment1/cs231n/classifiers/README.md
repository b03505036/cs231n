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

### Problem ((X-Y)^2)^(1/2)  
-> stuck in the size of input -> should be example 500X"3000" thank for [CSDN](https://blog.csdn.net/qq_22812319/article/details/79677525)