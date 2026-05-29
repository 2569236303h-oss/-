import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. 尝试导入 XGBoost
# =========================
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("提示：未安装 xgboost，将跳过 XGBoost 模型。")
    print("如需安装，请运行：pip install xgboost")

# =========================
# 2. 读取数据
# =========================
file_path = "student_behavior.csv"
df = pd.read_csv(file_path)

# 检查必要字段
required_cols = [
    "study_time",
    "practice_count",
    "homework_rate",
    "quiz_score",
    "participation",
    "learning_effect"
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"CSV 缺少必要字段：{missing_cols}")

# 如果没有样本编号，自动生成
if "样本编号" not in df.columns:
    df["样本编号"] = [f"S{i+1:03d}" for i in range(len(df))]

feature_cols = [
    "study_time",
    "practice_count",
    "homework_rate",
    "quiz_score",
    "participation"
]
target_col = "learning_effect"

X = df[feature_cols]
y = df[target_col]

# =========================
# 3. 划分训练集和测试集
# =========================
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, df.index,
    test_size=0.2,
    random_state=42
)

# =========================
# 4. 定义模型
# =========================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(
        random_state=42,
        max_depth=6,
        min_samples_leaf=3
    ),
    "随机森林": RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2
    )
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror"
    )

# =========================
# 5. 10折交叉验证
# =========================
cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_results = {}

print("10折交叉验证结果")
print("=" * 60)

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        },
        return_train_score=False
    )

    cv_rmse = -scores["test_rmse"].mean()
    cv_mae = -scores["test_mae"].mean()

    cv_results[name] = {
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae
    }

    print(f"{name:8s} -> RMSE: {cv_rmse:.4f}, MAE: {cv_mae:.4f}")

# =========================
# 6. 按交叉验证结果选择最优模型
# =========================
best_model_name = min(cv_results, key=lambda k: cv_results[k]["cv_rmse"])
best_model = models[best_model_name]

# 用训练集重新拟合最优模型
best_model.fit(X_train, y_train)

print("\n" + "=" * 60)
print(f"按交叉验证 RMSE 选择的最优模型：{best_model_name}")
print("=" * 60)

# =========================
# 7. 测试集评估
# =========================
print("\n测试集结果")
print("=" * 60)

test_results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    test_results[name] = {
        "rmse": rmse,
        "mae": mae
    }
    fitted_models[name] = model

    print(f"{name:8s} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

best_test_model_name = min(test_results, key=lambda k: test_results[k]["rmse"])
best_test_model = fitted_models[best_test_model_name]

print("\n" + "=" * 60)
print(f"按测试集 RMSE 选择的最优模型：{best_test_model_name}")
print("=" * 60)

# =========================
# 8. 输出测试集前10条预测结果
# =========================
y_test_pred = best_test_model.predict(X_test)

test_output = pd.DataFrame({
    "样本编号": df.loc[test_idx, "样本编号"].values,
    "真实值": y_test.values,
    "预测值": y_test_pred
})
test_output["误差"] = np.abs(test_output["真实值"] - test_output["预测值"])

print("\n测试集前10条预测结果：")
print(test_output.head(10).round(4))

# =========================
# 9. 个性化学习路径推荐
# =========================
def get_issues_and_recommendation(row):
    issues = []

    # 问题判断阈值
    if row["study_time"] < 3:
        issues.append("学习时长不足")
    if row["practice_count"] < 10:
        issues.append("练习不足")
    if row["homework_rate"] < 0.8:
        issues.append("作业完成较弱")
    if row["quiz_score"] < 70:
        issues.append("基础知识掌握不足")
    if row["participation"] < 0.4:
        issues.append("课堂参与较弱")

    # 推荐策略
    if not issues:
        return "整体表现较好", "保持优势 + 拓展训练 + 能力提升"

    if "学习时长不足" in issues:
        rec = "增加每日学习时间 + 制定周计划 + 番茄钟学习"
    elif "练习不足" in issues:
        rec = "增加每日练习 + 基础巩固 + 课后跟学"
    elif "作业完成较弱" in issues:
        rec = "按时完成作业 + 错题整理 + 及时订正"
    elif "课堂参与较弱" in issues:
        rec = "提升课堂参与 + 互动式学习 + 重点知识回顾"
    else:
        rec = "先补基础 + 例题精讲 + 分层练习"

    if len(issues) >= 2:
        rec += " + 阶段性复盘"

    return "、".join(issues), rec

# 对测试集做推荐
recommend_df = df.loc[test_idx, feature_cols].copy()
recommend_df["样本编号"] = df.loc[test_idx, "样本编号"].values
recommend_df["真实值"] = y_test.values
recommend_df["预测值"] = y_test_pred
recommend_df["主要问题"] = ""
recommend_df["推荐路径"] = ""

for idx, row in recommend_df.iterrows():
    issues, rec = get_issues_and_recommendation(row)
    recommend_df.at[idx, "主要问题"] = issues
    recommend_df.at[idx, "推荐路径"] = rec

# 按预测学习效果从低到高排序，取最需要干预的样本
recommend_df = recommend_df.sort_values(by="预测值", ascending=True)

print("\n个性化学习路径推荐前10条：")
show_cols = [
    "样本编号",
    "study_time",
    "practice_count",
    "homework_rate",
    "quiz_score",
    "participation",
    "真实值",
    "预测值",
    "主要问题",
    "推荐路径"
]
print(recommend_df[show_cols].head(10).round(4))

# =========================
# 10. 保存结果
# =========================
test_output.to_csv("test_predictions.csv", index=False, encoding="utf-8-sig")
recommend_df[show_cols].to_csv("recommendations.csv", index=False, encoding="utf-8-sig")

print("\n程序运行完成！")
print("已保存测试集预测结果：test_predictions.csv")
print("已保存推荐结果：recommendations.csv")
