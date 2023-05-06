plt记录一下loss
GPU内存没跑满，batch尝试往上调
增加层数

shuffle,优化器，batchsize,权重初始化

loss和准确率是否正相关

epoch会导致过拟合吗

加大or加深架构

## cnn10 
### CPU 10epoch
dev 47
test 80
### CPU 100epoch
dev 60
test 90

## cnn100
### CPU 10epoch
dev 29
test 39
### CPU_y7kp 10epoch
dev 23
test 25
### CPU_y7kp 100epoch
dev 29
test 30
### CPU_y7kp 50epoch
dev 27
test 32
### CPU_y7kp 40epoch
dev 22
test 29
出现了幺蛾子，两次都会在epoch=20左右loss突增
### CPU_y7kp 80epoch
dev 30
test 32
### CPU_y7kp 60epoch
dev 26
test 31
epoch=35时，loss就小于0.00000了
### GPU_y7kp 50epoch plus b=32
18 21
80epoch同
### GPU_y7kp 80epoch plus b=32 sgd
35 41
### GPU_y7kp 80epoch plus b=32 sgd
35 38
### GPU_y7kp 120epoch plus b=32 sgd
41 47 0.00222
### GPU_y7kp 200epoch plus b=32 sgd
41 50 0.00125
### GPU_y7kp 250epoch plus b=32 sgd
40 40 0.00087
### GPU_y7kp 100epoch plus b=64 sgd
32 35 0.01566
### GPU_y7kp 150epoch plus b=64 sgd
38 39 0.00534
### GPU_y7kp 50epoch plusplus b=16 sgd
33 35 0.00220
### GPU_y7kp 80epoch plusplus b=32 sgd
31 39 0.00289