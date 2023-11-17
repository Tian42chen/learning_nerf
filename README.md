# Learning NeRF

This repository is initially created by [Haotong Lin](https://haotongl.github.io/).

## Motivation of this repository

1. 面向实验室本科生的科研训练。通过复现NeRF来学习NeRF的算法、PyTorch编程。
2. 这个框架是实验室常用的代码框架，实验室的一些算法在这个框架下实现，比如：[ENeRF](https://github.com/zju3dv/enerf), [NeuSC](https://github.com/zju3dv/NeuSC), [MLP Maps](https://github.com/zju3dv/mlp_maps), [Animatable NeRF](https://github.com/zju3dv/animatable_nerf), [Neural Body](https://github.com/zju3dv/neuralbody)。实验室通过大量实践证明了这个代码框架的灵活性。学会使用这个框架，方便后续参与实验室的Project，也方便自己实现新的算法。

## Prerequisites

1. 确保你已经熟悉使用python, 尤其是debug工具：ipdb。

2. 计算机科学非常讲究自学能力和自我解决问题的能力，如果有一些内容没有介绍的十分详细，请先自己尝试探索代码框架。如果遇到代码问题，请先搜索网上的资料，或者查看仓库的Issues里有没有相似的已归档的问题。

3. 如果还是有问题，可以在这个仓库的issue里提问。

## Data preparation

Download NeRF synthetic dataset and add a link to the data directory. After preparation, you should have the following directory structure: 
```
data/nerf_synthetic
|-- chair
|   |-- test
|   |-- train
|   |-- val
|-- drums
|   |-- test
......
```


## 从Image fitting demo来学习这个框架


### 任务定义

训练一个MLP，将某一张图像的像素坐标作为输入, 输出这一张图像在该像素坐标的RGB value。

### Training

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

### Evaluation

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

### 查看loss曲线

```
tensorboard --logdir=data/record --bind_all
```


## 开始复现NeRF

### 配置文件

我们已经在configs/nerf/ 创建好了一个配置文件，nerf.yaml。其中包含了复现NeRF必要的参数。
你可以根据自己的喜好调整对应的参数的名称和风格。


### 创建dataset： lib.datasets.nerf.synthetic.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


#### debug：

```
python run.py --type dataset --cfg_file configs/img_fit/lego_view0.yaml
```

### 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。


#### debug：

```
python run.py --type network --cfg_file configs/img_fit/lego_view0.yaml
```

### loss模块和evaluator模块

这两个模块较为简单，不作仔细描述。

debug方式分别为：

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

## 花里胡哨的试探记录
### 问题整理
#### 背景处理
> 对 RGBA 图片, 如何处理出白色背景? 
```python
if cfg.task_arg.white_bkgd:
    image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
else:
    image = image[..., :3]
```
白色 RGB 为 (1, 1, 1), 所以当 `alpha = 0`, 结果就是白色. 

#### 坐标变换
> 网络不收敛, 可视化后发现是坐标范围不对
$$
x_c = R (x_w - c_x) = R x_w + T\\
T = -Rc_x
$$
$x_c$ 为 $x$ 相机坐标, $x_w$ 为世界坐标, $c_x$ 为相机在世界的坐标. 内参矩阵 (inner matrix) 一般都是 w2c, 世界到相机的. 

但是 nerf 的矩阵是转移矩阵 (transform matrix),  是 c2w, 相机到世界的, 也就是
$$
x_w = R^T x_c + c_x = R'x_c + T'
$$
所以求 rays 的时候坐标范围不对. 

#### `F.relu(rawalpha)`
对于 weights, 
```python
alpha = 1. - torch.exp(-F.relu(rawalpha)*dists)
weights = alpha * torch.cumprod(torch.cat([alpha.new_ones(alpha.shape[0], 1), 1.-alpha + epsilon], dim=-1), dim=-1)[..., :-1]
```
> 发现很多时候 `acc_map = torch.sum(weights, -1)` 全是 0. 

原因就是 `F.relu`, 当 `acc_map = 0`, 有 `alpha = 0`, 就有 `F.relu(rawalpha) = 0`, 然后有 `rawalpha<=0` . 所以一旦一个点被判断为背景, 或初始化时直接是负数, 其就不会再被激活, 因为 `relu` 不会传递负数的梯度. 大半的调试都是因为这个问题. 

##### 在 nerf pytorch 中
我猜测其使用了 pre crop 来减少初始时网络采样时一个点被判断为背景的几率, 以此对抗这个问题. 

> 这里遇到网络训练时有重影

原因是 pre crop 没有按照预期, 之前的记录方式不对, 导致从头到尾都是 crop, 所以训练的只有小图像, 外部没有信息, 当然有重影了. 

在代码框架之中, 因为使用 DataLoader 多线程加载 dataset (num workers>0) 时, 每个工作进程都会创建自己的 Python 解释器和内存空间, 这意味着每个工作进程都有自己的全局变量, 且每次 epoch 会使用刚初始化完成的 dataset 的状态.  

所以在多次尝试后, 我选择 IO 文件来记录 dataset 迭代的次数, 以此来控制 pre crop 的行为.

##### 我的解决方案
非常暴力, 强行减少初始化时直接是负数的几率. 

pytorch 默认 `nn.Linear` 初始化 `weights` 与 `bias`, 是使用 $\mathcal{U}(-\frac{1}{\sqrt{{W}}}, \frac{1}{\sqrt{{W}}})$, 我直接使用 $\mathcal{U}(-\frac{0.5}{\sqrt{{W}}}, \frac{1}{\sqrt{{W}}})$, 让其平均偏向于正数. 

对训练结果的影响未知. 




### 原始记录
- [x] ~~对白色点的采样可能有问题.~~ 是 `volume rendering` 没有设置对, 没用 `bg_brightness = 1`
- [x] 粗细 nerf, 看着似乎是直接用两个网络, 再看了下, 关键是 `sample_pdf` 函数. 
- [x] 但网络不收敛, 应该是我的问题, 首先是单采样也不收敛, 所以要确认 network 所有行为都符合预期. 现在粗细收敛了, 似乎在降低又似乎没在降x
  - [x] 头都要裂开, nerf 的 transform matrix 并不是 extrinsic matrix(w2c), 其定义是 c2w. 所以  T 直接是 相机的世界坐标. 
  - [x] 破防了, 不只这个问题. 没想到问题真的还是 `sample_pdf`, 分层采了个寂寞的样. 修好了. 
  - 有希望
- [x] 没希望, 估计还是数据集出问题了, 或者某步转换出问题了. 训练时好时坏的. 以下是遇到的问题
  - [x] 有坏点, 这个应该是因为初始化的问题, 导致一部分网络是无法训练的. 
  - [x] ~~粗网络不收敛, 最后只有细网络有数据.~~ ~~粗网络直接寄了, 应该是有其他问题~~ 这是因为初始化的问题
  - [x] rawalpha 是负的, 然后 relu 一下就没数值了
    - [x] 可以尝试训一波, 看看寄的情况下 rawalpha 是不是就不正了. 
    - rawalpha 会是负的, 然后过 relu, 就会得出全是 0 的 alpha
    - 现在有两个方案:
      1. 修改初始化
         1. 修改 weights 的: 输入的 x 不可能是负的 (因为 relu), 所以只需要让期望稍微偏移即可
         2. 修改 bias 的: ~~感觉非常的行, 不过需要一定的数值, 不然还是有负的.~~ 其实还是不行, bias 只加一次, 因为负的 weights, 最后还是非常容易负. 
      2. 修改 relu 为 leaky_relu
    - 最后选择方案 1, 修改 weights, bias 采样区间为 $\mathcal{U}(-\frac{0.5}{\sqrt{{W}}}, \frac{1}{\sqrt{{W}}})$
    - 最后没改, 因为 nerf pytorch 似乎也没处理这个问题. 最后改了, nerf pytorch 的 precrop 应该是针对这个问题 (的吧)
- [x] 现在的问题是粗网络下面会有重影, 应该需要多看看其他视角, 确认是什么原因导致的重影
  - [ ] 先把 vis 给写了吧
  - [x] wc, 换了 get rays, rays_d 就没有 norm 了, 所以加了个 rays_d 的 norm. 
  - [x] 还是不太行, 还是有重影. 算了, 改成和 nerf pytroch 的处理一样的吧
  - [x] 离谱, `np.random.choice(..., replace = False)` 与 `np.random.randint` 是不一样的, 后者会重复选择. 
  - [x] 还是不行, 看看 rgb 有没有问题
- [x] 呃, 和 nerf py 不一样的地方是它 500 precrop 后会有个 acc & loss 的 跳降, 就一步就跳降了, 真的离谱. (因为我的 crop 没有达到预期, 所以没有跳降)
  - [x] 想看看初始化的时候其 acc 是啥样的, 看了, 没看出啥
- [x] 算了, 摆了, 准备 cv 一份代码, 然后替换成自己的部分再调一调
  - [x] batch 不太对, 逝逝. 很逝. 
  - [x] 有用, 但背景还是很脏. 应该还是 weight 的问题. (crop没有达到预期
- [x] 再次尝试合并网络. dataset, renderer, network, 0 为 我, 1 为 别人. 以 1000 iter (2 epoch) 为基准.
  0. 111: 好的
  1. 011, 没有 precrop: 好的
  2. 011, precrop: 不行, acc 近乎为 1, 没有跳降. 
  3. 000, 没有 precrop: 失败的有点多, 感觉太过容易坍缩为 acc=0 了. 不过成功的情况下还行. 
  4. 原版实现, 没有 precrop: 失败的也有点多. 不过成功情况也没有问题. 
- 离大谱, 是 precrop 的问题, 统计次数的东西没有生效. 
  - 产生使用 class, 不行, dataset 创建实例会拷贝这个 class, 跟我想象的指针不太一样, 因为 python 的垃圾回收机制估计也不太好改. 
  - 四个 dataset 之间的全局变量不共享 ( ﾟ∀。)
  - 解决啦!!!!!!!!!!
- [x] 到时候整理下这个记录x
  - 所以我调了两周的 nerf 其实就遇到 4 个比较大的问题? 主要是定位 pre crop 这个问题花的时间太多了. 