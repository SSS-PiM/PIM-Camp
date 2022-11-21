# PIM-Camp

服务于大二学生，尽量讲的细一点

### week0 - 环境搭建

#### Vscode + WSL

* [安装 vscode](https://code.visualstudio.com/)
* vscode 安装 Remote-SSH 和 WSL 插件
* [安装 WSL](https://zhuanlan.zhihu.com/p/386590591)

#### C++ Depencency

安装好 WSL 后，使用 vscode 连接到 WSL，执行以下命令

```shell
sudo apt-get update
sudo apt-get install gcc
sudo apt-get install g++
sudo apt-get install build-essential
```

#### Git

* 生成秘钥

  ```shell
  ssh-keygen -t rsa
  ```
* 注册 Github 账号

  https://github.com/
* 添加公钥

  在 Settings 里面添加公钥

  ```shell
  cat ~/.ssh/id_rsa.pub
  ```

  执行这条命令，将打印出来的公钥复制到 Github 上，添加。

#### NeuroSim

* 拉代码

  ```shell
  git clone git@github.com:neurosim/MLP_NeuroSim_V3.0.git
  ```

### week1 - 神经网络
- #### [Tutorial](./week1/DNN_Tutorial.md)

- #### [Lecture Note](./week1/DNN_Tutorial_LN.md)

### [week2 - 加速器简介](./week2/加速器.md)

### [week3 - MNSIM简介](./week3/mnsim.md)

### [week4 - NeuroSim简介](./week4/Neurosim.pptx)

### [week5 - 实验说明书](./week5/Instruction.pdf)