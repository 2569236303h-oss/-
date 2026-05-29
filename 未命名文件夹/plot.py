import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. 中文字体设置
# =========================
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv("student_behavior.csv")
test_pred = pd.read_csv("test_predictions.csv")

# =========================
# 3. learning_effect 分布图
# =========================
plt.figure(figsize=(8, 5))
sns.histplot(df["learning_effect"], bins=20, kde=True, color="#4C72B0")
plt.title("learning_effect 分布图")
plt.xlabel("learning_effect")
plt.ylabel("频数")
plt.tight_layout()
plt.savefig("learning_effect_distribution.png", dpi=300)
plt.show()

# =========================
# 4. 各模型 RMSE 柱状图
# =========================
models = ["线性回归", "决策树", "随机森林", "XGBoost"]
cv_rmse = [0.0815, 0.1079, 0.0842, 0.0824]
test_rmse = [0.0770, 0.0934, 0.0755, 0.0772]

rmse_df = pd.DataFrame({
    "模型": models,
    "交叉验证RMSE": cv_rmse,
    "测试集RMSE": test_rmse
})

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, rmse_df["交叉验证RMSE"], width=width, label="交叉验证RMSE", color="#55A868")
plt.bar(x + width/2, rmse_df["测试集RMSE"], width=width, label="测试集RMSE", color="#C44E52")

plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("各模型 RMSE 对比")
plt.legend()
plt.tight_layout()
plt.savefig("model_rmse_comparison.png", dpi=300)
plt.show()

# =========================
# 5. 真实值 vs 预测值散点图
# =========================
plt.figure(figsize=(7, 7))
plt.scatter(test_pred["真实值"], test_pred["预测值"], alpha=0.75, color="#8172B2")

# 参考线 y=x
min_val = min(test_pred["真实值"].min(), test_pred["预测值"].min())
max_val = max(test_pred["真实值"].max(), test_pred["预测值"].max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值散点图")
plt.tight_layout()
plt.savefig("true_vs_pred_scatter.png", dpi=300)
plt.show()

print("图表已保存：")
print("learning_effect_distribution.png")
print("model_rmse_comparison.png")
print("true_vs_pred_scatter.png")
