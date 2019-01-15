

# Q1: how fasttext classify text? how to get word vector?

## 1. 处理输入、构造词典
file: dictionary.cc
function: readFromFile
详细过程见 dictionary.h 中的注释

## 2. 构造模型

模型结构:  
inputs -- hidden -- output

## 3. 训练模型 (loss_name: softmax)

### supervised/cbow/skipgram 模型的区别

- 输出层不同
    - supervised 以所有 label 的预测概率作为输出
    - cbow 和 skipgram 以所有 word 的预测概率作为输出
- 样本选择方式不同
    - supervised 以 (一行文本, 一个label) 作为单个样本，更新模型
    - cbow 以一行文本中的 (Context(w), w) 作为单个样本，更新模型
    - skipgram 以一行文本中的 (w, Context(w)[i]) 作为单个样本，更新模型

> 单个样本以 (inputs, target) 表示，inputs 为 hidden 的输入，其本身是一个由 wid 组成的序列；target 表示该 inputs 序列要预测的目标  
> Context(w) : 每次在 [1, args-ws] 中随机一个窗口大小 win_size，取 w 周围 win_size 个词作为 Context(w)  
> Context(w)[i] : Context(w) 里面的第 i 个词  

### 单个样本对模型的更新过程

**符号表**:

- IN: 输入向量矩阵，IN(i) 表示第 i 个 word 的向量，维度为 nwords x dim
- OUT: 输出向量矩阵，OUT(i) 表示第 i 个 target 的向量。对于 supervised 模型，OUT 的维度为 nlabels x dim；对于 cbow 和 skipgram 模型，OUT 的维度为 nwords x dim
- hidden: 维度为 dim x 1
- output: 维度为 ntargets x 1

**更新过程**:

1. calc hidden  
hidden = inputs 向量加和，再按 inputs 大小求平均
2. calc output  
output = OUT dot hidden = ntargets x dim dot dim x 1 = ntargets x 1
3. update gradient and OUT  
```
for i in ntargets:
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]); // q1, 见下
    gradient += alpha * OUT(i)              // q2
    OUT(i) += alpha * hidden                // q3

```
4. gradient 按 inputs 大小求平均
5. update IN  
```
for i in ninputs:
    IN(inputs(i)) += gradient               // q3

```

## q1: 损失函数应该怎么写？为什么可以这么计算梯度？

损失函数是交叉熵(cross-entropy)  
梯度的计算过程见[Softmax 函数的特点和作用是什么？ - 忆臻的回答 - 知乎](https://www.zhihu.com/question/23765351/answer/240869755)

## [TODO] q2: q1 行的代码不是用来计算梯度的么，q2 又是什么?

## [TODO] q3: 为什么可以这么更新 OUT 和 IN?

# Q2: 不同 loss_function 分别适用什么场景?

## softmax 

文本分类，nlabels << nwords

## hs: hierarchy softmax

softmax 的复杂度 O(n)。hierarchy softmax 的复杂度 O(log(n))  
当输出层 size 越大，hierarchy softmax 对性能的优化越明显。

## [TODO] ns: negative sampling

