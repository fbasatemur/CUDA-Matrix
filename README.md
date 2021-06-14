# CUDA-Matrix

## 2D Convolution
The 3x3 kernel mask do convolution on the 2D matrix.For RxC dimensional input, (R-2)x(C-2) dimensional output matrix is created.

<img src="https://i.stack.imgur.com/WoNbd.gif" alt="alt text" width="575" height="400">

## 3D Convolution
The 3x3x3 kernel mask do convolution on the 3D matrix. For RxCxD dimensional input, (R-2)x(C-2) dimensional output matrix is created.

<img src="https://miro.medium.com/max/3220/1*fnkPPcqdWOtgg0ewoEfr9g.png" alt="alt text" width="575" height="400">

## Matrix Multiplication
**Matrix x Vector** or **Matrix x Matrix** multiplication can be performed in parallel
```
* NxM x MxK = NxK       // matrix x matrix = matrix
* 1xM x MxK = 1xK       // vector x matrix = vector
* NxN x NxN = NxN       // square_Matrix x square_Matrix = S_Matrix
```
<img src="https://thumbs.gfycat.com/PositiveExhaustedAmericangoldfinch-small.gif" alt="alt text" width="575" height="400">
