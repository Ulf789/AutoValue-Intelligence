# model_engine.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_predict(user_model, user_year, user_mileage):
    # 1. 读入我们在第一步造好的“历史账本”
    df = pd.read_csv('cars.csv')
    
    # 2. 划分“因果关系”
    # X 是“原因”（特征）：车型、年份、里程
    X = df[['model', 'year', 'mileage']]
    # y 是“结果”（标签）：我们要预测的价格
    y = df['price']
    
    # 3. 处理文本数据（独热编码 One-Hot Encoding）
    # 电脑只认识数字，不认识 "BMW M5" 这样的字符串。
    # ColumnTransformer 会自动把文本车型变成电脑能看懂的 0 和 1 矩阵。
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['model'])],
        remainder='passthrough'
    )
    
    # 4. 组装“流水线 (Pipeline)”与选择算法
    # 我们选择“随机森林回归算法 (RandomForestRegressor)”，它非常擅长处理二手车这种多特征的估值。
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=30, random_state=42))
    ])
    
    # 5. 开启训练（让算法疯狂看账本，寻找折旧规律）
    pipeline.fit(X, y)
    
    # 6. 接收用户输入，进行估值
    # 把你在网页上选的车型、年份、里程包装成和账本一模一样的格式
    input_data = pd.DataFrame([[user_model, user_year, user_mileage]], columns=['model', 'year', 'mileage'])
    
    # 让训练好的精算师出单子
    predicted_price = pipeline.predict(input_data)[0]
    
    # 返回整数价格
    return int(predicted_price)

# 7. 本地快速测试：看看精算师及不及格
if __name__ == "__main__":
    print("正在测试机器学习估值引擎...")
    # 测试一辆 2022 年、开了 40000 公里的宝马 M5 应该卖多少钱
    test_price = train_and_predict("BMW M5", 2022, 40000)
    print(f"模拟测试结果：一辆 2022 款开了 40000 km 的 BMW M5，AI 预测基准价为: {test_price} EUR")