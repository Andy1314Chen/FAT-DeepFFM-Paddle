## FAT-DeepFFM

### 一、简介

本项目是基于 PaddleRec 框架对 FAT-DeepFFM CTR 预估算法进行复现。

论文：[FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/abs/1905.06336)

![FAT-DeepFFM 网络结构](https://tva1.sinaimg.cn/large/008i3skNly1gtn0ys467uj610i0u0tc102.jpg)

该模型是 FM 系列，属于 FM 的复杂变种，相比基础的 FM 算法，有 3 处改动：
- 增加 Field 向量概念，变成 **F**FM 
- FM 特征交叉后，经 DNN 增强泛化性，变成 **Deep**FFM
- 参考 CV 领域特征提取网络 SENet, 在 Embedding Layer 和 FM Layer 之间增加 CENet, 进行特征筛选(去其糟粕，取其精华), 变成了 **FAT**-DeepFFM

结合上面 3 处改动，就可以很容易看清 FAT-DeepFFM 的网络结构及其学习步骤了：

- sparse features 经由 Embedding Layer 查表得到 embedding 矩阵（这里不是向量了，因为增加了 Feature Field 的概念，每个 Field 都会对应一个向量）
- 上述特征向量，经过 Attentive Embedding Matrix Layer。作者类似 SENet, 提出了一个 CENet 进行特征筛选，增强有用特征，弱化噪声
- 处理好的特征输入 Feature Interaction Layer 进行特征交叉(就是 FFM 特征交叉，但比 FM 慢了好多。。。)
- FFM 特征交叉后，经过 MLP 再次增强模型泛化性，输出预测概率

还有个小改动，在进行 FMM 特征交叉时， 作者认为 Hadamard Product 要比 Inner Product 效果好。前者是输出一维向量，而点积得到的是标量，其实前者按维度求和
就是点积。

![](https://tva1.sinaimg.cn/large/008i3skNly1gtn1f69hlej61dy0a8gn402.jpg)

### 二、复现精度

![](https://tva1.sinaimg.cn/large/008i3skNly1gtn1istjtfj611p0u0n4102.jpg)

上图中，`I` 代表了 FFM 中是使用 inner product, `H`就表示使用 hadamard product.

本次 PaddlePaddle 论文复现赛要求在 Criteo 数据集上，FAT-DeepFFM 的复现精度为 AUC > 0.8099. 

实际本项目复现精度为：

### 三、数据集

原论文采用 Kaggle Criteo 数据集，为常用的 CTR 预估任务基准数据集。单条样本包括 13 列 dense features、 26 列 sparse features及 label.

[Kaggle Criteo 数据集](https://www.kaggle.com/c/criteo-display-ad-challenge)
- train set: 4584, 0617 条
- test set:   604, 2135 条 （no label)

[PaddleRec Criteo 数据集](https://github.com/PaddlePaddle/PaddleRec/blob/release/2.1.0/datasets/criteo/run.sh)
- train set: 4400, 0000 条
- test set:   184, 0617 条

不过作者对 Criteo 数据集进行了随机处理，按照 9：1 重新划分训练集和测试集，本项目遵循该部分操作。因此，训练集与测试集数据如下：
- train set: 4125, 6555 条
- test set:   458, 4061 条

![数据集划分介绍](https://tva1.sinaimg.cn/large/008i3skNly1gtn1wdgt6rj616a0c042g02.jpg)

P.S. Criteo 原始数据集是存在时序关系的，理论上为了避免数据穿越，应该将 last day 数据作为测试集的。但本项目复现过程中遵循原论文相同数据预处理。


### 四、环境依赖
- 硬件：CPU、GPU
- 框架：
  - PaddlePaddle >= 2.1.2
  - Python >= 3.7

### 五、快速开始

该小节操作建议在百度 AI-Studio NoteBook 中进行执行。

AIStudio 项目链接：[https://aistudio.baidu.com/studio/project/partial/verify/2281174/3987013dd88e45ce828d3b9a3f2d24a9](https://aistudio.baidu.com/studio/project/partial/verify/2281174/3987013dd88e45ce828d3b9a3f2d24a9), 可以 fork 一下。

#### 1. AI-Studio 快速复现步骤
(约 6 个小时，可以直接在 notebook 切换版本加载预训练模型文件)

```
################# Step 1, git clone code ################
# 当前处于 /home/aistudio 目录, 代码存放在 /home/work/rank/DIFM-Paddle 中

import os
if not os.path.isdir('work/rank/DIFM-Paddle'):
    if not os.path.isdir('work/rank'):
        !mkdir work/rank
    !cd work/rank && git clone https://hub.fastgit.org/Andy1314Chen/DIFM-Paddle.git

################# Step 2, download data ################
# 当前处于 /home/aistudio 目录，数据存放在 /home/data/criteo 中

import os
os.makedirs('data/criteo', exist_ok=True)

# Download  data
if not os.path.exists('data/criteo/slot_test_data_full.tar.gz') or not os.path.exists('data/criteo/slot_train_data_full.tar.gz'):
    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_test_data_full.tar.gz
    
    !cd data/criteo && wget https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
    !cd data/criteo && tar xzvf slot_train_data_full.tar.gz

################## Step 3, train model ##################
# 启动训练脚本 (需注意当前是否是 GPU 环境）
!cd work/rank/DIFM-Paddle && sh run.sh config_bigdata

```

#### 2. criteo slot_test_data_full 验证集结果
```
...
2021-08-14 11:53:10,026 - INFO - epoch: 0 done, auc: 0.799622, epoch time: 261.84 s
2021-08-14 11:57:32,841 - INFO - epoch: 1 done, auc: 0.799941, epoch time: 262.81 s
```

#### 3. 使用预训练模型进行预测
- 在 notebook 中切换到 `V1.2调参数` 版本，加载预训练模型文件，可快速验证测试集 AUC；
- ！！注意 config_bigdata.yaml 的 `use_gpu` 配置需要与当前运行环境保存一致 
```
!cd /home/aistudio/work/rank/DIFM-Paddle && python -u tools/infer.py -m models/rank/difm/config_bigdata.yaml
```

### 六、代码结构与详细说明

代码结构遵循 PaddleRec 框架结构
```
|--models
  |--rank
    |--fat-deepffm            # 本项目核心代码
      |--data                 # 采样小数据集
      |--config.yaml          # 采样小数据集模型配置
      |--config_bigdata.yaml  # Kaggle Criteo 全量数据集模型配置
      |--criteo_reader.py     # dataset加载类         
      |--download_data.sh     # 数据下载及训练集划分脚本  
      |--dygraph_model.py     # PaddleRec 动态图模型训练类
      |--net.py               # fat-deepffm 核心算法代码
|--tools                      # PaddleRec 工具类
      |--train_and_eval.py    # 对 PaddleRec 框架进行了修改，支持 train 和 infer 交替执行(trainer.py 和 infer.py 源码也进行了微改动）
      |--...
|--LICENSE                    # 项目 LICENSE
|--README.md                  # readme
|--run.sh                     # 项目执行脚本(需在 aistudio notebook 中运行)
```