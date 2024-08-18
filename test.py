import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.random.randint(0, 60, (100, 100))

# 创建绘图
plt.figure(figsize=(6, 6))
plt.imshow(data, cmap='gray', interpolation='none')
plt.colorbar()  # 添加颜色条
plt.title("Clean")  # 设置标题
plt.grid(visible=True, color="black", linewidth=0.5)  # 显示网格

# 显示图像
plt.show()
