import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "outputs"
for directory in (FIG_DIR, OUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    math_df = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")
    math_df["subject"] = "math"
    por_df = pd.read_csv(BASE_DIR / "student-por.csv", sep=";")
    por_df["subject"] = "portuguese"
    full = pd.concat([math_df, por_df], ignore_index=True)
    return full


def build_features(df: pd.DataFrame):
    target = "G3"
    drop_columns = ["G3", "G1", "G2"]
    feature_df = df.drop(columns=drop_columns)
    categorical_cols = feature_df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [column for column in feature_df.columns if column not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return feature_df, df[target], preprocessor


def evaluate_models(X, y, preprocessor):
    models = {
        "Ridge": Ridge(alpha=2.0, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=4,
        ),
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted_models = {}
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        cv_scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append(
            {
                "model": name,
                "cv_rmse_mean": -np.mean(cv_scores["test_rmse"]),
                "cv_rmse_std": np.std(-cv_scores["test_rmse"]),
                "cv_mae_mean": -np.mean(cv_scores["test_mae"]),
                "cv_r2_mean": np.mean(cv_scores["test_r2"]),
            }
        )

        fitted_models[name] = pipeline

    result_df = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    return result_df, fitted_models


def holdout_prediction(X, y, best_model: Pipeline):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
    }
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "error": y_pred - y_test.values,
        }
    ).reset_index(drop=True)
    return metrics, pred_df


def export_feature_importance(best_model: Pipeline):
    model = best_model.named_steps["model"]
    preprocessor = best_model.named_steps["preprocess"]
    if not hasattr(model, "feature_importances_"):
        return None
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(OUT_DIR / "xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")
    return imp_df


def learning_path_bucket(score: float) -> str:
    if score < 10:
        return "基础提升路径（知识补齐+作业陪练+节奏管理）"
    if score < 14:
        return "巩固强化路径（错题循环+专题训练+课堂互动）"
    if score < 17:
        return "进阶拓展路径（综合题组+跨学科阅读+项目实践）"
    return "卓越挑战路径（高阶问题+竞赛题+同伴辅导）"


def export_recommendations(pred_df: pd.DataFrame):
    sample = pred_df.sample(12, random_state=RANDOM_STATE).copy().reset_index(drop=True)
    sample["student_id"] = [f"S{i:03d}" for i in range(1, len(sample) + 1)]
    sample["path"] = sample["y_pred"].apply(learning_path_bucket)
    sample = sample[["student_id", "y_true", "y_pred", "error", "path"]]
    sample.to_csv(OUT_DIR / "sample_recommendations.csv", index=False, encoding="utf-8-sig")
    return sample


def save_plots(df: pd.DataFrame, result_df: pd.DataFrame, pred_df: pd.DataFrame):
    sns.set_theme(style="whitegrid", font="Arial Unicode MS")
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df["G3"], bins=20, kde=True, color="#1f77b4")
    plt.title("图1 真实数据中最终成绩 G3 分布")
    plt.xlabel("G3 (0-20)")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_g3_distribution.png", dpi=300)
    plt.close()

    corr_cols = ["G3", "studytime", "failures", "absences", "Medu", "Fedu", "goout", "Walc", "Dalc"]
    plt.figure(figsize=(8, 6))
    corr_df = df[corr_cols].corr(numeric_only=True)
    sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True)
    plt.title("图2 关键数值特征相关性热力图")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_correlation_heatmap.png", dpi=300)
    plt.close()

    rank_df = result_df.sort_values("cv_rmse_mean", ascending=True)
    plt.figure(figsize=(8, 4.8))
    sns.barplot(data=rank_df, x="model", y="cv_rmse_mean", hue="model", palette="viridis", legend=False)
    plt.title("图3 各模型10折交叉验证 RMSE 对比（越低越好）")
    plt.xlabel("模型")
    plt.ylabel("CV RMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_model_rmse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    sns.scatterplot(x="y_true", y="y_pred", data=pred_df, alpha=0.7, color="#2ca02c")
    min_v = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    max_v = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], color="#d62728", linestyle="--", linewidth=1.5)
    plt.title("图4 测试集真实值与预测值对比")
    plt.xlabel("真实值 G3")
    plt.ylabel("预测值 G3")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_actual_vs_pred.png", dpi=300)
    plt.close()


def main():
    df = load_data()
    X, y, preprocessor = build_features(df)
    result_df, fitted_models = evaluate_models(X, y, preprocessor)
    best_model_name = result_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    holdout_metrics, pred_df = holdout_prediction(X, y, best_model)
    importance_df = export_feature_importance(best_model)
    sample_df = export_recommendations(pred_df)
    save_plots(df, result_df, pred_df)

    result_df.to_csv(OUT_DIR / "model_cv_results.csv", index=False, encoding="utf-8-sig")
    pred_df.head(40).to_csv(OUT_DIR / "holdout_predictions_head40.csv", index=False, encoding="utf-8-sig")

    report = {
        "dataset_rows": int(len(df)),
        "dataset_cols": int(df.shape[1]),
        "best_model": best_model_name,
        "holdout_metrics": holdout_metrics,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    numeric_desc = df[["G3", "studytime", "failures", "absences", "Medu", "Fedu"]].describe().round(3)
    numeric_desc.to_csv(OUT_DIR / "numeric_describe.csv", encoding="utf-8-sig")

    print("DONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n示例推荐：")
    print(sample_df.head(6).to_string(index=False))
    if importance_df is not None:
        print("\n关键特征：")
        print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
