from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 生成训练数据
X, y = load_iris(return_X_y=True)
# 只取两类进行二分类任务
X = X[y != 2]
y = y[y != 2]

# 训练数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 初始化参数
n_features = X_train.shape[1]
theta = np.random.randn(n_features, 1)
bias = np.random.randn(1)

# 学习率
lr = 0.01
epochs = 1000

# 模型运算函数
def forward(x, theta, bias):
    # 线性运算
    z = np.dot(x, theta) + bias
    # sigmoid函数
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat

# 损失函数
def loss(y, y_hat):
    e = 1e-8
    return -y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 计算梯度
def calc_gradient(x, y, y_hat):
    m = x.shape[0]  # 注意这里的维度应该是样本数量
    delta_theta = np.dot(x.T, (y_hat - y).reshape(-1, 1)) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias

# 模型训练
for i in range(epochs):
    # 前向计算
    y_hat = forward(X_train, theta, bias)
    # 计算损失
    loss_val = loss(y_train, y_hat.flatten())
    # 计算梯度
    delta_theta, delta_bias = calc_gradient(X_train, y_train, y_hat.flatten())
    # 更新参数
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias

    if i % 100 == 0:
        # 计算准确率
        acc = np.mean(np.round(y_hat.flatten()) == y_train)
        print(f"epoch:{i}, loss:{np.mean(loss_val)},acc:{acc}")
