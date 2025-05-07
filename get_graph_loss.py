import matplotlib.pyplot as plt

# 整理后的数据 (epoch, loss, iter_time)
data = [
    (200, 0.01438757, 0.14292),
    (400, 0.00842368, 0.14336),
    (600, 0.04302338, 0.14316),
    (800, 0.03454389, 0.14506),
    (1000, 0.0067131, 0.14408),
    (1200, 0.06391037, 0.14095),
    (1400, 0.00561859, 0.14239),
    (1600, 0.04026299, 0.14318),
    (1800, 0.00693905, 0.14319),
    (2000, 0.0065163, 0.13910),  # 使用后一次2000的数据
    (4000, 0.00478315, 0.14398),
    (6000, 0.01927107, 0.14866),
    (8000, 0.00654437, 0.14826),
    (10000, 0.0181518, 0.15056),
    (12000, 0.02005606, 0.14411),
    (14000, 0.01844629, 0.14227),
    (16000, 0.01717822, 0.14546),
    (18000, 0.01367294, 0.14523),
    (20000, 0.01231308, 0.14467),
    (22000, 0.00338119, 0.14506),
]

# 拆分数据
epochs = [d[0] for d in data]
losses = [d[1] for d in data]
times = [d[2] for d in data]



# 绘制 Loss vs Epoch
plt.figure()

# 设置刻度字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.plot(epochs, losses, marker='o', color='blue')
plt.title("SSIM Loss vs Epoch",fontsize=16)
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("SSIM Loss",fontsize=16)
plt.grid(True)
plt.show()

# 绘制 Iteration Time vs Epoch
plt.figure()

# 设置刻度字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.plot(epochs, times, marker='x', color='green')
plt.title("Iteration Time vs Epoch",fontsize=16)
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Iteration Time (s)",fontsize=16)
plt.grid(True)
plt.show()
# import matplotlib.pyplot as plt

# # 手动整理的数据（epoch, loss, iter_time）
# data = [
#     (200, 0.01438757, 0.14292),
#     (400, 0.00842368, 0.14336),
#     (600, 0.04302338, 0.14316),
#     (800, 0.03454389, 0.14506),
#     (1000, 0.0067131, 0.14408),
#     (1200, 0.06391037, 0.14095),
#     (1400, 0.00561859, 0.14239),
#     (1600, 0.04026299, 0.14318),
#     (1800, 0.00693905, 0.14319),
#     (2000, 0.0065163, 0.13910),  # 保留后面的这条
#     (4000, 0.00478315, 0.14398),
#     (6000, 0.01927107, 0.14866),
#     (8000, 0.00654437, 0.14826),
#     (10000, 0.0181518, 0.15056),
#     (12000, 0.02005606, 0.14411),
#     (14000, 0.01844629, 0.14227),
#     (16000, 0.01717822, 0.14546),
#     (18000, 0.01367294, 0.14523),
#     (20000, 0.01231308, 0.14467),
#     (22000, 0.00338119, 0.14506),
# ]

# # 拆分数据
# epochs = [d[0] for d in data]
# losses = [d[1] for d in data]
# times = [d[2] for d in data]

# # 画图
# plt.figure(figsize=(12, 5))

# # 子图1：损失函数
# plt.subplot(1, 2, 1)
# plt.plot(epochs, losses, marker='o', color='blue')
# plt.title("Loss vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)

# # 子图2：迭代时间
# plt.subplot(1, 2, 2)
# plt.plot(epochs, times, marker='x', color='green')
# plt.title("Iteration Time vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Iteration Time (s)")
# plt.grid(True)

# plt.tight_layout()
# plt.show()


