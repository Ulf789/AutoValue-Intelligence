import pandas as pd
import numpy as np

# 设置随机种子，确保你和我的数据结果一致
np.random.seed(42)
n_samples = 500

print("正在模拟德国二手车市场数据...")

# 1. 随机生成 500 辆车的车型、首次上牌年份、当前里程数
models = np.random.choice(["BMW M5", "3 Series Touring", "Audi R8"], n_samples)
years = np.random.randint(2016, 2026, n_samples)
mileages = np.random.randint(5000, 150000, n_samples)

# 2. 设定每种车型的市场初始基准价（新车或极新车的价格）
base_prices = {"BMW M5": 90000, "3 Series Touring": 35000, "Audi R8": 130000}
prices = []

# 3. 根据汽车市场客观规律，用数学公式计算出每辆车的合理二手售价
for m, y, mil in zip(models, years, mileages):
    p = base_prices[m]
    
    # 规律 A：年份越新越贵。以 2016 年为基准，每年轻一岁，价格加 4000 欧
    p += (y - 2016) * 4000  
    
    # 规律 B：里程越多越便宜。每多行驶 1000 公里，折旧 150 欧
    p -= (mil / 1000) * 150  
    
    # 规律 C：加入市场随机波动（噪声），比如车况好坏、原车主加装的选配等
    p += np.random.normal(0, 3000)  
    
    # 确保价格不会变成负数，最低设定为 10000 欧
    prices.append(max(int(p), 10000))

# 4. 把这些数据打包成一个标准的电子表格格式（DataFrame）
df = pd.DataFrame({
    "model": models,
    "year": years,
    "mileage": mileages,
    "price": prices
})

# 5. 导出保存为本地 CSV 文件
df.to_csv("cars.csv", index=False)
print("✅ 成功！‘历史账本’已生成，文件名叫 `cars.csv`，里面包含 500 条高性能二手车市场数据！")