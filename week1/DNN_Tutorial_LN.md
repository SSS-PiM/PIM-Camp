### Deep Neural Network - Tutorial

#### **Lecture Note**

##### Task 1:

（1）使用Linear激活无法收敛，使用Sigmoid激活可以收敛；

（2）平方特征本身是非线性的，且符合所用数据本身的分布；因此神经网络不需要再提取非线性特征，只需要通过线性组合即可对数据进行很好的分类；综合前两问，这里是希望同学们理解，对于具有非线性特征分布的数据进行分类，其本质是提取特征、组合特征，最终反映数据的分布；提取和组合既可以在神经网络外完成，也可以在神经网络内部完成；在本例中，在网络外完成了所需特征的提取后，即便神经网络使用Linear函数激活，失去了非线性特征的提取能力，依然能很好地完成分类任务。这说明分类的本质是一种特征的提取和拟合。

（3）使用ReLU较使用Sigmoid取得了显著更快的收敛速度，但ReLU取得的决策边界也不如Sigmoid取得的平滑。

##### Task 2:

梯度更新表达式如下：
$$
\begin{align}
	\frac{\partial L}{\partial w_2} &= \underbrace{\frac{\partial L}{\partial z_2}}_{\delta_2} \frac{\partial z_2}{\partial w_2} \\
	\frac{\partial L}{\partial w_1} &= \underbrace{\frac{\partial L}{\partial z_2} \frac{\partial z_2}{\partial a_1} \frac{\partial a_1}{\partial z_1}}_{\delta_1} \frac{\partial z_1}{\partial w_1}
\end{align}
$$

其中：
$$
\begin{align}
	\delta_2 &= \frac{\partial L}{\partial z_2} \\
	\delta_1 &= \delta_2 \frac{\partial z_2}{\partial a_1} \frac{\partial a_1}{\partial z_1}
\end{align}
$$
通常Loss函数选择交叉熵，配合之，输出层使用softmax函数，设输出为$\hat{y}$，实际标签为$y$，则交叉熵损失为：
$$
L(y, \hat{y}) = -y^Tlog\hat{y}
$$
 softmax函数输出表示为：
$$
\begin{align}
	\hat{y} &= softmax(z_{opt}) \\
		\hat{y_i}	&= \frac{e^{z_{opt}^{i}}}{\sum_k e^{z_{opt}^k}}
\end{align}
$$
给出ReLU函数和Sigmoid函数的表达式：
$$
\begin{align}
ReLU(x) &= max(0, x)\\
Sigmoid(x) &= \frac{1}{1+e^{-x}}
\end{align}
$$
则**重点**在解决下面四个求导问题，**难点**在下括号标出的三个部分：
$$
\begin{align}
	\frac{\partial L}{\partial z_2} & = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z_2} \\ 
	 &= \underbrace{-\frac{\partial }{\partial \hat{y}}\left( y^Tlog\hat{y}\right)}_{(1)} \space \underbrace{\frac{\partial}{\partial z_2}\left( softmax(z_2)\right)}_{(2)} \tag{a} \\
	 \frac{\partial a_{l}}{\partial z_{l}} &= \underbrace{\frac{\partial \sigma{(z_{l})}}{\partial z_{l}}}_{(3)} \tag{b} \\
     \frac{\partial z_{l+1}}{\partial a_{l}} &= w_{l+1} \tag{c} \\
	 \frac{\partial z_{l+1}}{\partial w_{l+1}} &= a_{l} \tag{d}
\end{align}
$$
考虑$(1)$：
$$
\begin{align}
	-\frac{\partial}{\partial \hat{y}}\left( y^Tlog\hat{y}\right) &= -y^T 
	\begin{bmatrix}
		\frac{\partial \log \hat{y_1}}{\partial\hat{y_1}} & \frac{\partial \log \hat{y_1}}{\partial\hat{y_2}} & \cdots & \frac{\partial \log \hat{y_1}}{\partial\hat{y_n}} \\
		\frac{\partial \log \hat{y_2}}{\partial\hat{y_1}} & \frac{\partial \log \hat{y_2}}{\partial\hat{y_2}} & \cdots & \frac{\partial \log \hat{y_2}}{\partial\hat{y_n}} \\
		\vdots & \vdots & \ddots & \vdots \\
				\frac{\partial \log \hat{y_n}}{\partial\hat{y_1}} & \frac{\partial \log \hat{y_n}}{\partial\hat{y_2}} & \cdots & \frac{\partial \log \hat{y_n}}{\partial\hat{y_n}} \\
	 \end{bmatrix} \\
	 &= -y^T
	 	\begin{bmatrix}
		\frac{1}{\hat{y_1}} & 0 & \cdots & 0 \\
		0 & \frac{1}{\hat{y_2}} & \cdots & 0 \\
		\vdots & \vdots & \ddots & \vdots \\
		0 & 0 & \cdots & \frac{1}{\hat{y_n}} \\
	 	\end{bmatrix} \\
	 &= -\begin{bmatrix}
	 \frac{y_1}{\hat{y_1}} & \frac{y_2}{\hat{y_2}} & \cdots & \frac{y_n}{\hat{y_n}}
	 \end{bmatrix} \\
	 &= -\begin{bmatrix}
	 0 & \cdots & 0 & \frac{y_i}{\hat{y_i}} & 0 & \cdots & 0
	 \end{bmatrix} \Bigg|_{y_i = 1} \\
\end{align}
$$
考虑$(2)$：
$$
\begin{align}
	\frac{\partial}{\partial z}\left(softmax(z)\right) &= 
	\begin{bmatrix}
		\frac{\partial}{\partial z_1}\left(\frac{e^{z_1}}{\sum_k e^{z_k}}\right) & \frac{\partial}{\partial z_2}\left(\frac{e^{z_1}}{\sum_k e^{z_k}}\right) & \cdots & \frac{\partial}{\partial z_n}\left(\frac{e^{z_1}}{\sum_k e^{z_k}}\right) \\
		\frac{\partial}{\partial z_1}\left(\frac{e^{z_2}}{\sum_k e^{z_k}}\right) & \frac{\partial}{\partial z_2}\left(\frac{e^{z_2}}{\sum_k e^{z_k}}\right) & \cdots & \frac{\partial}{\partial z_n}\left(\frac{e^{z_2}}{\sum_k e^{z_k}}\right) \\
		\vdots & \vdots & \ddots & \vdots \\
		\frac{\partial}{\partial z_1}\left(\frac{e^{z_n}}{\sum_k e^{z_k}}\right) & \frac{\partial}{\partial z_2}\left(\frac{e^{z_n}}{\sum_k e^{z_k}}\right) & \cdots & \frac{\partial}{\partial z_n}\left(\frac{e^{z_n}}{\sum_k e^{z_k}}\right) \\
	\end{bmatrix} \\ \\
	&= \space \cdots \cdots\\ \\
	&= \begin{bmatrix}
	\hat{y_1}(1-\hat{y_1}) & -\hat{y_1} \hat{y_2} & \cdots & -\hat{y_1} \hat{y_n} \\
	-\hat{y_2} \hat{y_1} & \hat{y_2}(1-\hat{y_2}) & \cdots & -\hat{y_2} \hat{y_n} \\
	\vdots & \vdots & \ddots & \vdots \\
	-\hat{y_n} \hat{y_1} & \hat{y_n} \hat{y_2} & \cdots & \hat{y_n} (1-\hat{y_n}) \\
	\end{bmatrix}
\end{align}
$$
 综合则有：
$$
\begin{align}
	\frac{\partial L}{\partial z_2} & = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z_2} \\ 
	 &= -\frac{\partial }{\partial \hat{y}}\left( y^Tlog\hat{y}\right)\space \frac{\partial}{\partial z_2}\left( softmax(z_2)\right) \tag{a} \\
	 &=
	\begin{bmatrix}
		\hat{y}_1 & \cdots & \hat{y}_{i-1} & \hat{y}_i - 1 & \hat{y}_{i+1} & \cdots & \hat{y}_n
	\end{bmatrix}
\end{align}
$$
考虑$(3)$，若$\sigma$为$ReLU(\cdot{})$，则：
$$
\begin{align}
	\frac{\partial a_{l}}{\partial z_{l}} &= \frac{\partial \sigma{(z_{l})}}{\partial z_{l}} \tag{b} \\
	&=diag(\mathbb I(z_l^{(i)} > 0))
\end{align}
$$
若$\sigma$为$Sigmoid(\cdot{})$，则：
$$
\begin{align}
	\frac{\partial a_{l}}{\partial z_{l}} &= \frac{\partial \sigma{(z_{l})}}{\partial z_{l}} \tag{b} \\
	&=diag_i(\sigma(z_l^{(i)})(1-\sigma(z_l^{(i)}))) \\ \\
	As \space there's &\space property:\\
	&\sigma^{\prime}(\cdot) = \sigma(\cdot) (1- \sigma(\cdot))\\
	when \space \sigma \space is \space &Sigmoid(\cdot).
	
\end{align}
$$

##### Appendix A: 万能近似定理

> Hornik, K., Stinchcombe, M., and White, H. (1989). Multilayer feedforward networks are uni-versal approximators. Neural Networks, 2, 359–366. 171

**万能近似定理（Universal Approximation Theorem）**表明，一个前馈神经网络如果具有线性输出层和至少一层具有任何一种“挤压”性质的激活函数的隐藏层，只要给予网络足够数量的隐藏单元，**它可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的Borel可测函数**。

[直观理解万能近似定理(Universal Approximation theorem) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/443284394)

[万能近似定理 - 腾讯云开发者社区](https://cloud.tencent.com/developer/article/1454579)

##### Appendix B: Convolutional Neural Network

卷积神经网络基本原理：

> Zhang, W., 1988. Shift-invariant pattern recognition neural network and its optical architecture. In Proceedings of annual conference of the Japan Society of Applied Physics.
>
> LeCun, Y. and Bengio, Y., 1995. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10), 1995.
>
> LeCun, Y., Boser, B., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W. and Jackel, L.D., 1989. Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4), pp.541-551.

卷积神经网络在图像分类任务中的应用：

> C. Szegedy et al., "Going deeper with convolutions," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1-9, doi: 10.1109/CVPR.2015.7298594.
>
> Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2017. ImageNet classification with deep convolutional neural networks. Commun. ACM 60, 6 (June 2017), 84–90. https://doi.org/10.1145/3065386