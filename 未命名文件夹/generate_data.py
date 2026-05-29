import pandas as pd
import numpy as np

# =========================
# 1. 固定随机种子
# =========================
np.random.seed(2021)

# =========================
# 2. 生成样本数量
# =========================
n_samples = 300

# =========================
# 3. 生成基础特征
# =========================
data = pd.DataFrame({
    "study_time": np.random.uniform(0, 10, n_samples),         # 学习时长：0~10小时
    "practice_count": np.random.randint(0, 50, n_samples),     # 练习次数：0~49
    "homework_rate": np.random.uniform(0.3, 1.0, n_samples),   # 作业完成率：0.3~1.0
    "quiz_score": np.random.uniform(40, 100, n_samples),       # 测验分数：40~100
    "participation": np.random.uniform(0, 1, n_samples)        # 课堂参与度：0~1
})

# =========================
# 4. 特征标准化（用于构造学习效果）
# =========================
study_z = (data["study_time"] - 5) / 3.0
practice_z = (data["practice_count"] - 25) / 14.0
homework_z = (data["homework_rate"] - 0.65) / 0.2
quiz_z = (data["quiz_score"] - 70) / 15.0
participation_z = (data["participation"] - 0.5) / 0.25

# =========================
# 5. 构造原始学习效果分数
#    加入一些交互项和噪声，让数据更真实
# =========================
raw_score = (
    0.30 * study_z +
    0.20 * practice_z +
    0.25 * homework_z +
    0.20 * quiz_z +
    0.15 * participation_z +
    0.12 * study_z * homework_z +
    0.10 * practice_z * participation_z -
    0.08 * (1 - data["homework_rate"]) * (1 - data["participation"]) +
    np.random.normal(0, 0.35, n_samples)
)

# =========================
# 6. 使用 sigmoid 映射到 0~1
#    避免大量样本被截成 1.0
# =========================
data["learning_effect"] = 1 / (1 + np.exp(-raw_score))

# 稍微拉开分布范围，避免太集中
data["learning_effect"] = 0.05 + 0.9 * data["learning_effect"]

# 保留四位小数
data["learning_effect"] = data["learning_effect"].round(4)

# =========================
# 7. 保存数据
# =========================
output_file = "student_behavior.csv"
data.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================
# 8. 检查输出
# =========================
print("=" * 50)
print("数据已生成并保存为：", output_file)
print("=" * 50)

print("\n【数据基本信息】")
print(data.info())

print("\n【learning_effect 分布统计】")
print(data["learning_effect"].describe())

print("\n【learning_effect 前10个值】")
print(data["learning_effect"].head(10).to_list())

print("\n【前5行数据】")
print(data.head())

print("\n【是否存在全为1.0的情况】")
print("最大值是否为1.0：", float(data["learning_effect"].max()) == 1.0)
print("最小值：", float(data["learning_effect"].min()))

print("\n【前10行检查】")
print(data[["study_time", "practice_count", "homework_rate", "quiz_score", "participation", "learning_effect"]].head(10))
