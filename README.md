# Machine Learning Notes

## HandWriting
- [HandWriting.py](HandWriting/HandWriting.py)

- Input: 
  - 四维张量形状: 样本数量、图像高度、图像宽度、通道数
  - 通过除以 255.0 将像素值归一化到 0 到 1 的范围

- Build Model:
    - 使用 Sequential 模型构建一个简单的 CNN 模型。
        - 卷积层（32 个过滤器、3x3 大小、ReLU 激活函数）
            - 提取图像中的特征。卷积层通过应用一组可学习的滤波器（也称为卷积核或特征检测器）在输入图像上进行滑动窗口操作，从而计算每个滤波器的卷积响应。这样可以捕捉到不同位置的局部特征，例如边缘、纹理等。
            - 激活函数引入非线性性质，允许 CNN 模型学习更复杂的函数关系。常用的激活函数包括ReLU（Rectified Linear Unit）、Sigmoid 和 Tanh。ReLU 是最常用的激活函数，因为它在计算上更高效并且能够缓解梯度消失问题。
        - 一个最大池化层（2x2 大小）
            - 在每个池化区域中选择最大的特征响应值作为输出。池化层有助于减少参数数量、减轻计算负担，并且对于平移和缩放的图像具有一定程度的不变性。
        - 一个展平层
            - 假设我们有一个特征图的形状为 (batch_size, height, width, channels)，通过展平层，它将被展开为形状为 (batch_size, height * width * channels) 的一维向量。
        - 两个全连接层（64 个神经元、ReLU 激活函数和 10 个神经元、softmax 激活函数）
            - 用于分类或回归任务。全连接层将前一层的输出连接到每个神经元，每个神经元对应一个类别或回归输出。

- Compile Model:
    - adam 优化器
        - 更新模型的权重和偏差，以使损失函数最小化。常用的优化算法包括随机梯度下降（Stochastic Gradient Descent，SGD）和其变种（如 Adam、Adagrad 等），用于根据损失函数的梯度调整模型参数。
    - sparse_categorical_crossentropy 损失函数
        - 衡量预测结果与真实标签之间的差异。常用的损失函数包括交叉熵损失函数（Cross-Entropy Loss）用于分类任务，均方误差损失函数（Mean Squared Error Loss）用于回归任务。
    - 评估指标 准确率。

- Train Model: 
    - 训练的轮数（epochs）
    - 批量大小（batch_size）
    - 验证集

- Evaluate Model:
     - 损失值
     - 准确率
