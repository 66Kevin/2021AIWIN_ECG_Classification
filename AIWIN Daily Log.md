# AIWIN比赛日记

1. focalLoss改成了BCELoss后，loss更稳定，训练速度提高，反向传播更快，

2. 将人工统计特征放入深度模型内一起训练，会有效果提升

3. 本来训练集数据量就不大，如果分出训练集的最后20%数据作验证集，用于测试是否保存模型参数的话会造成浪费，但是全部拿来训练的话又不知道什么时候该保存模型参数??

   1. **Q**: 交叉验证会得到多个模型，在预测数据时，是把多个模型的预测值加总求平均值吗？

      **A**: 交叉验证一般并**不是**用于训练模型进行预测，而是求出多次的评估分数，把评估分数求均值, 来**确定模型的效果**。
      如果用这个思想，得到多个模型，然后把预测值求均值，那就是“集成学习”了，并不算是交叉验证啦。

   2. **Q**: 对于交叉验证部分，不说K-fold的K是多少，那么最后会产生K个模型，一般来说，每次训练出来的模型的参数w都是不一样的，所以最后既然已经训练完成了，那么我应该选取那个模型呢？
   
      **A**: **交叉验证并不是为了去获取一个模型，他的主要目的是为了去评估这个模型的一些特性，**所以说，你这个最后要选取的模型，这些个都不是，而应该是去采用全部数据来训练出来的模型。
   
   3. **Q**: 那如何利用交叉验证选取最优的模型呢？
   
      **A**: 借鉴https://zhuanlan.zhihu.com/p/83841282中的回答：
   
      先在training set中选取一小部分当作test set，再从剩余的training set中分成K折进行交叉验证（即K-1用来训练，K折用来验证），虽然training set分成了K折但是每一折都会经历训练的过程，并且保证每一折在验证时，都能保证模型使用没有见过的数据进行验证模型效果。保存下来每一折中最优模型（即val最高的那个epoch的模型），记录下来模型大概的训练轮次，拿交叉验证中得到的经验来训练整个training set。最后用Test set来测试每折最优的模型与整个训练集得到的模型，选最优的模型即可。
   
      <img src="/Users/kevin/Library/Application Support/typora-user-images/IMG_1858.jpg" alt="IMG_1858" style="zoom:25%;" />

4. 或者使用早停来确定保存最优模型参数：
   早停需要验证集，这意味着某些样本不能用于模型的训练过程，这会造成数据的浪费。为了更好地利用验证集的样本，可以在早停之后进行额外的训练。在第二轮额外的训练中，所有的训练数据都被包括在内（包括验证集)
   
   1. 保留迭代步：再次初始化模型，然后使用所有数据再次训练，此时使用第一轮早停确定的最佳步数作为第二轮的迭代步数。
   2. 保留参数：保持从第一轮训练中获得的参数，然后使用全部的数据继续训练。此时观察原始验证集的损失西数，直到它低于
      第一轮停止时的原始训练集的损失函数值
   
4. 我们不光可以将同一份数据喂入多个模型，也可以将多份相似数据喂入同一个模型，这也是一种常用的集成学习技巧，称为测试集数据扩增(Test Time Augmentation，简称TTA)。也就是说，**数据扩增不仅可以在训练时候用，而且可以同样在预测时候
   进行数据扩增，对同一个样本预测多次，然后对多次结果进行平均**。
   
4. 模型蒸馏：大模型训练完之后，用训练集预测出的标签用于小模型的训练

4. 保存loss和accury数据尽量用dict

   ```python
       history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
       for epoch in range(num_epochs):
           train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
           test_loss, test_correct=valid_epoch(model,device,test_loader,criterion)
   
           train_loss = train_loss / len(train_loader.sampler)
           train_acc = train_correct / len(train_loader.sampler) * 100
           test_loss = test_loss / len(test_loader.sampler)
           test_acc = test_correct / len(test_loader.sampler) * 100
   
           print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} 					%".format(epoch + 1, num_epochs,train_loss,test_loss,train_acc,test_acc))
           
           history['train_loss'].append(train_loss)
           history['test_loss'].append(test_loss)
           history['train_acc'].append(train_acc)
           history['test_acc'].append(test_acc)
   ```

8. BN-ReLu-ConV效果比Conv-BN-ReLu好

9. Transformer爆显存问题

   ```python
   RuntimeError: CUDA out of memory. Tried to allocate 192.00 MiB (GPU 0; 31.75 GiB total capacity; 29.69 GiB already allocated; 55.50 MiB free; 30.39 GiB reserved in total by PyTorch)
   ```

    解决方案：

   1. 查看某个时间点显存占用的状态：用显存观察代码（电路表）掐在每一个能使得显存变化的位置上，比如将model.cuda()，tensor.to(device)的前后进行监控

      ```python
      import pynvml
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
      print(meminfo.used)
      ```

   2. 分析模型：发现，显存爆炸的地方出现在了句子length达到了4960的长度，远远大于最大长度512. Transformer由多层的encoder组成，恐怖的地方在于多层transformer的可怕规模的计算图。也就是说如果句子长度大于512这个数量级，那么即使是4的batch size都在每一次构建model计算图时吃掉10+GB的显存

   反思：

   1. 深入了解pytorch机制，什么会影响显存？大型计算图中，像transformer，哪里容易累积梯度和cache？

7. transformer在数据量大的时候才有用！！！

8. 时间序列特征构造

   1. **时域特征**：主要是根据 ECG 信号的 RR 间期得到的统计特征，包括 RR 间期的最大值 （maxRR）、最小值（minRR）、概率密度估计值等特征，还有心率的中位数（medHR）、相 邻 RR 间期差值（ΔRR）的均方根（RMSSD）及 ΔRR 间期大于 50ms 的百分比（PNN50） 等。由于房颤信号的 RR 间期绝对不规则，因此这些特征在区分正常信号和房颤信号时能够 发挥较好的效果。

   2. **频域特征**：主要是对ECG信号加窗，计算窗口信号的谱参数，包括谱质心、质心频率等， 同时还计算了小波变换系数、归一化低频功率和归 一化高频功率

   3. **非线性特征**：主要计算 Lorenz plot 相关参数、Poincare plot 参数、Hjorth 参数等，和 绝对中位差（median absolute deviation，MAD），即三个相邻 RR 间期心率平均值的绝对偏 差变化率的中位数，同时包括样本熵（sample entropy，SampEn）、近似熵（approximate entropy，ApEn）、模糊测量熵（fuzzy entropy，FuzzyEn）及归一化模糊熵（normalized fuzzy entropy，NFEn）等熵值。

   4. **形态特征**：主要是根据 P 波、Q 波、R 波、S 波和 T 波位置及幅值，判断 ECG 波形的变 形情况，主要计算 S 波和 Q 波相对 R 波的深度、ST 段斜率、QT 间期及 QRS 波群宽度等

      <img src="/Users/kevin/Library/Application Support/typora-user-images/image-20211211164810092.png" alt="image-20211211164810092" style="zoom:50%;" />

9. 如何挖掘时间和空间信息？
10. 能否构造新的特征？如果不构造新的特征，要如何从给出的特征中挖掘
    出更多的信息？
11. 训练集的数据量不大，如何增加数据量？如何构造小（参数量小，减小过拟合风险）而深（能提取出足够丰富的信息）的模型？
12. 如何划分训练集和验证集，保证验证集与测试集的分布尽可能一致？
13. 如何提高模型的长距离预测能力？
