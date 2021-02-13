# CUDA-Matrix

## 2D Convolution
The 3x3 kernel mask do convolution on the 2D matrix.For RxC dimensional input, (R-2)x(C-2) dimensional output matrix is created.

<img src="https://i.stack.imgur.com/WoNbd.gif" alt="alt text" width="500" height="350">

## 3D Convolution
The 3x3x3 kernel mask do convolution on the 3D matrix. For RxCxD dimensional input, (R-2)x(C-2) dimensional output matrix is created.

<img src="https://media-exp1.licdn.com/dms/image/C4D12AQGF5Z6WwEHKYg/article-inline_image-shrink_1000_1488/0/1550737338775?e=1618444800&v=beta&t=U1rfN6o7tQf6nlVVSmx_-y-_w9Hv2rpu-TMoxdiXOMw" alt="alt text" width="500" height="350">

## Matrix Multiplication
**Matrix x Vector** or **Matrix x Matrix** multiplication can be performed in parallel
```
* NxM x MxK = NxK       // matrix x matrix = matrix
* 1xM x MxK = 1xK       // vector x matrix = vector
* NxN x NxN = NxN       // square_Matrix x square_Matrix = S_Matrix
```
<img src="https://thumbs.gfycat.com/PositiveExhaustedAmericangoldfinch-small.gif" alt="alt text" width="500" height="350">
