# Loss-aware-weight-quantization
Implementation of ICLR 2018 paper "Loss-aware Weight Quantization of Deep Networks", tested with python 2.7, theano 0.9.0 and lasagne 0.2.dev1.


This repository is divided in two subrepositories:

- FNN: enables the reproduction of the FNN results(on MNIST, CIFAR-10, CIFAR-100, SVHN) reported in the article

- RNN: enables the reproduction of the RNN results(on War and Peace, Linux Kernel, PTB) reported in the article

Requirements
This software is implemented on top of the implementation of [BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect) and has all the same requirements. 


Example training command  on *War and Peace* dataset:
- training using approximate ternarization method LATa
```sh
python warpeace.py --method="LATa" --lr_start=0.002  --len=100
```
- training using 3-bit linear quantization method LAQ_linear
```sh
python warpeace.py --method="LAQ_linear" --lr_start=0.002  --len=100
```
