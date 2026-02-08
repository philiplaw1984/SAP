
# 综合版本：集成roc3.py的所有模型，应用roc4_2.py的双提升优化策略，加入传统评分比较

# 综合版本：集成roc3.py的所有模型，应用roc4_2.py的双提升优化策略，加入传统评分比较

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, \
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve, accuracy_score,
    recall_score, confusion_matrix, f1_score, fbeta_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import os
from datetime import datetime
from scipy.special import expit
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ========== 导入XGBoost ==========
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost未安装，将跳过XGBoost模型")
    XGB_AVAILABLE = False


# ========== 自定义分类器：双提升优化器 ==========
class DualOptimizerClassifier(BaseEstimator, ClassifierMixin):
    """同时优化recall和precision的分类器"""

    def __init__(self, base_model, alpha=0.5, optimize_method='f2_pr', n_thresholds=101):
        self.base_model = base_model
        self.alpha = alpha
        self.optimize_method = optimize_method
        self.n_thresholds = n_thresholds
        self.optimal_threshold = 0.5
        self.is_fitted = False
        self.base_estimator_ = None

    def fit(self, X, y, sample_weight=None):
        """训练并优化"""
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        # 克隆基础模型
        self.base_estimator_ = clone(self.base_model)

        # 训练基础模型
        if sample_weight is not None:
            self.base_estimator_.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.base_estimator_.fit(X_train, y_train)

        # 获取验证集概率
        if hasattr(self.base_estimator_, 'predict_proba'):
            y_val_proba = self.base_estimator_.predict_proba(X_val)[:, 1]
        elif hasattr(self.base_estimator_, 'decision_function'):
            decision_scores = self.base_estimator_.decision_function(X_val)
            y_val_proba = expit(decision_scores)
        else:
            y_val_proba = None

        # 优化阈值
        if y_val_proba is not None:
            self.optimal_threshold = self._optimize_threshold(y_val, y_val_proba)

        # 在整个训练集上重新训练
        if sample_weight is not None:
            self.base_estimator_.fit(X, y, sample_weight=sample_weight)
        else:
            self.base_estimator_.fit(X, y)

        self.is_fitted = True
        return self

    def _optimize_threshold(self, y_true, y_pred_proba):
        """优化阈值以同时提高recall和precision"""
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_score = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            if precision == 0 or recall == 0:
                continue

            if self.optimize_method == 'f2_pr':
                # 方法1: F2分数 + PR AUC的组合
                f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                # 计算局部PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                local_pr_auc = auc(recall_curve, precision_curve) if len(recall_curve) > 1 and len(
                    precision_curve) > 1 else 0
                score = 0.6 * f2 + 0.4 * local_pr_auc

            elif self.optimize_method == 'geometric':
                # 方法2: 几何平均 + 调和平均
                geometric_mean = np.sqrt(precision * recall)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                score = 0.5 * geometric_mean + 0.5 * f1

            elif self.optimize_method == 'custom':
                # 方法3: 自定义权重，考虑alpha参数
                score = (self.alpha * precision + (1 - self.alpha) * recall) * np.sqrt(precision * recall)

            else:
                # 默认方法: F2.5分数 (偏recall但不过分)
                score = fbeta_score(y_true, y_pred, beta=2.5, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        if hasattr(self.base_estimator_, 'predict_proba'):
            return self.base_estimator_.predict_proba(X)
        elif hasattr(self.base_estimator_, 'decision_function'):
            decision_scores = self.base_estimator_.decision_function(X)
            proba_positive = expit(decision_scores)
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            raise AttributeError("基础模型没有predict_proba或decision_function方法")

    def get_params(self, deep=True):
        return {
            'base_model': self.base_model,
            'alpha': self.alpha,
            'optimize_method': self.optimize_method,
            'n_thresholds': self.n_thresholds
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ========== 自定义分类器：传统评分分类器 ==========
class TraditionalScoreClassifier(BaseEstimator, ClassifierMixin):
    """传统评分分类器"""

    def __init__(self, score_type='APACHEII', optimal_threshold=0.5):
        self.score_type = score_type
        self.optimal_threshold = optimal_threshold
        self.is_fitted = False
        self.score_values = None
        self.score_min = None
        self.score_max = None

    def fit(self, X, y):
        # 传统评分不需要训练，但为了接口统一
        # 假设X包含传统评分列
        if self.score_type in X.columns:
            self.score_values = X[self.score_type].values
            self.score_min = self.score_values.min()
            self.score_max = self.score_values.max()
        else:
            # 如果找不到对应列，尝试模糊匹配
            for col in X.columns:
                if self.score_type.lower() in col.lower():
                    self.score_values = X[col].values
                    self.score_min = self.score_values.min()
                    self.score_max = self.score_values.max()
                    break

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("模型未训练")

        # 获取评分值
        if self.score_type in X.columns:
            score_vals = X[self.score_type].values
        else:
            # 尝试模糊匹配
            score_vals = None
            for col in X.columns:
                if self.score_type.lower() in col.lower():
                    score_vals = X[col].values
                    break

            if score_vals is None:
                print(f"警告: 未找到{self.score_type}评分列，使用0.5作为默认概率")
                score_vals = np.ones(len(X)) * 0.5

        # 归一化到0-1范围
        if self.score_max > self.score_min:
            score_norm = (score_vals - self.score_min) / (self.score_max - self.score_min)
        else:
            score_norm = np.ones_like(score_vals) * 0.5

        # 确保在0-1之间
        score_norm = np.clip(score_norm, 0, 1)

        # 返回两列概率矩阵
        return np.column_stack([1 - score_norm, score_norm])

    def get_params(self, deep=True):
        return {'score_type': self.score_type, 'optimal_threshold': self.optimal_threshold}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ========== 传统评分评估函数 ==========
def evaluate_traditional_score(score_type, score_values, y_true, n_folds=30):
    """评估传统评分的性能"""

    # 确保评分值是numpy数组
    score_values = np.array(score_values)

    # 归一化到0-1范围
    score_min, score_max = score_values.min(), score_values.max()
    if score_max > score_min:
        score_norm = (score_values - score_min) / (score_max - score_min)
    else:
        score_norm = np.ones_like(score_values) * 0.5

    # 设置交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 存储结果
    fold_results = {
        'pr_auc': [], 'roc_auc': [], 'precision': [],
        'recall': [], 'f1': [], 'f_beta': [], 'balanced_score': [],
        'optimal_threshold': [], 'predictions': [], 'probabilities': []
    }

    fold = 0
    for train_idx, test_idx in skf.split(score_norm.reshape(-1, 1), y_true):
        fold += 1
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        score_train, score_test = score_norm[train_idx], score_norm[test_idx]

        # 在训练集上优化阈值
        thresholds = np.linspace(0.1, 0.9, 81)
        best_score = -1
        optimal_threshold = 0.5

        for threshold in thresholds:
            y_pred = (score_train >= threshold).astype(int)
            precision = precision_score(y_train, y_pred, zero_division=0)
            recall = recall_score(y_train, y_pred, zero_division=0)

            if precision > 0 and recall > 0:
                f2 = fbeta_score(y_train, y_pred, beta=2, zero_division=0)
                geometric_mean = np.sqrt(precision * recall)
                f1_val = f1_score(y_train, y_pred, zero_division=0)
                balanced_score = 0.4 * f2 + 0.3 * precision + 0.2 * recall + 0.1 * geometric_mean + 0.1 * f1_val

                if balanced_score > best_score:
                    best_score = balanced_score
                    optimal_threshold = threshold

        # 使用优化后的阈值进行预测
        y_pred = (score_test >= optimal_threshold).astype(int)
        y_pred_proba = score_test

        # 计算指标
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

        # PR AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals) if len(recall_vals) > 1 and len(precision_vals) > 1 else np.mean(
            y_test)

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr) if len(fpr) > 1 and len(tpr) > 1 else 0.5

        # 平衡分数（使用roc4_2中的定义）
        geometric_mean = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
        balanced_score = 0.4 * precision + 0.4 * recall + 0.2 * f1

        # 存储结果
        fold_results['pr_auc'].append(pr_auc)
        fold_results['roc_auc'].append(roc_auc)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        fold_results['f_beta'].append(f2)
        fold_results['balanced_score'].append(balanced_score)
        fold_results['optimal_threshold'].append(optimal_threshold)
        fold_results['predictions'].append(y_pred)
        fold_results['probabilities'].append(y_pred_proba)

    # 计算平均指标
    mean_results = {}
    std_results = {}
    for key in fold_results:
        if key not in ['predictions', 'probabilities']:
            mean_results[f'mean_{key}'] = np.mean(fold_results[key])
            std_results[f'std_{key}'] = np.std(fold_results[key])

    return {
        'fold_results': fold_results,
        'mean_results': mean_results,
        'std_results': std_results,
        'score_values': score_values,
        'score_norm': score_norm
    }


# ========== 数据加载和预处理 ==========
print("=" * 80)
print("加载和预处理数据...")
print("=" * 80)
data = pd.read_csv('SAP653_ARDS.csv')
np.set_printoptions(suppress=True)
y = data["ARDS"].values

# 检查数据中是否包含传统评分
print("\n检查数据集中是否包含传统评分列...")
traditional_score_cols = []
traditional_score_data = {}

for col in data.columns:
    col_lower = col.lower()
    if 'apache' in col_lower or 'apacheii' in col_lower:
        print(f"  找到APACHEII评分列: {col}")
        traditional_score_cols.append(('APACHEII', col))
        traditional_score_data['APACHEII'] = data[col].values
    elif 'ranson' in col_lower:
        print(f"  找到Ranson评分列: {col}")
        traditional_score_cols.append(('Ranson', col))
        traditional_score_data['Ranson'] = data[col].values
    elif 'bisap' in col_lower:
        print(f"  找到BISAP评分列: {col}")
        traditional_score_cols.append(('BISAP', col))
        traditional_score_data['BISAP'] = data[col].values

# 分离特征和传统评分
feature_cols = []
traditional_score_feature_cols = []  # 存储传统评分列名
for col in data.columns:
    if col != "ARDS":
        is_traditional_score = False
        for score_type, score_col in traditional_score_cols:
            if col == score_col:
                is_traditional_score = True
                traditional_score_feature_cols.append(col)
                break
        if not is_traditional_score:
            feature_cols.append(col)

# 获取特征数据（不包含传统评分）
X_original = data[feature_cols].values  # 原始特征数据，用于机器学习模型

print(f"\n数据集信息:")
print(f"  总样本数: {len(y)}")
print(f"  特征维度: {X_original.shape[1]} (排除传统评分)")
print(f"  正类(ARDS)数量: {sum(y)} ({np.mean(y) * 100:.2f}%)")
print(f"  负类数量: {len(y) - sum(y)} ({(1 - np.mean(y)) * 100:.2f}%)")

# 计算正类比例
pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0
print(f"  正类权重比例: {pos_weight:.2f}")

if traditional_score_cols:
    print(f"\n找到的传统评分列:")
    for score_type, col_name in traditional_score_cols:
        score_values = traditional_score_data[score_type]
        print(
            f"  {score_type}: 列名='{col_name}', 范围=[{score_values.min():.2f}, {score_values.max():.2f}], 均值={score_values.mean():.2f}±{score_values.std():.2f}")

# 保存完整的数据集（用于传统评分模型）
X_with_scores_full = data.copy()  # 包含所有列，包括传统评分

# ========== 评估传统评分（如果存在） ==========
traditional_score_results = {}
if traditional_score_cols:
    print(f"\n{'=' * 80}")
    print("评估传统评分性能...")
    print("=" * 80)

    for score_type, col_name in traditional_score_cols:
        score_values = traditional_score_data[score_type]
        results = evaluate_traditional_score(score_type, score_values, y, n_folds=30)
        traditional_score_results[score_type] = results

        # 打印结果
        print(f"\n{score_type}评分性能:")
        print(
            f"  PR AUC:          {results['mean_results']['mean_pr_auc']:.4f} ± {results['std_results']['std_pr_auc']:.4f}")
        print(
            f"  ROC AUC:         {results['mean_results']['mean_roc_auc']:.4f} ± {results['std_results']['std_roc_auc']:.4f}")
        print(
            f"  精确率:          {results['mean_results']['mean_precision']:.4f} ± {results['std_results']['std_precision']:.4f}")
        print(
            f"  召回率:          {results['mean_results']['mean_recall']:.4f} ± {results['std_results']['std_recall']:.4f}")
        print(f"  F1分数:          {results['mean_results']['mean_f1']:.4f} ± {results['std_results']['std_f1']:.4f}")
        print(
            f"  F2分数:          {results['mean_results']['mean_f_beta']:.4f} ± {results['std_results']['std_f_beta']:.4f}")
        print(
            f"  平衡分数:        {results['mean_results']['mean_balanced_score']:.4f} ± {results['std_results']['std_balanced_score']:.4f}")

# ========== 模型初始化 ==========
print(f"\n{'=' * 80}")
print("初始化所有模型...")
print("=" * 80)

models = {}

# ========== 第1组：线性模型（带双提升优化） ==========
models["Dual_Logistic_Regression"] = DualOptimizerClassifier(
    base_model=LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    alpha=0.5,
    optimize_method='f2_pr'
)

# ========== 第2组：支持向量机 ==========
models["Dual_SVM_RBF"] = DualOptimizerClassifier(
    base_model=SVC(
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    alpha=0.5,
    optimize_method='f2_pr'
)

# LinearSVC需要特殊处理，因为它没有predict_proba
linear_svc = LinearSVC(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)
models["Linear_SVM"] = linear_svc  # 保持原样，不包装

# ========== 第3组：决策树 ==========
models["Dual_Decision_Tree"] = DualOptimizerClassifier(
    base_model=DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ),
    alpha=0.5,
    optimize_method='geometric'
)

# ========== 第4组：随机森林家族 ==========
models["Dual_Random_Forest"] = DualOptimizerClassifier(
    base_model=RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    alpha=0.4,
    optimize_method='geometric'
)

models["Dual_Extra_Trees"] = DualOptimizerClassifier(
    base_model=ExtraTreesClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    alpha=0.4,
    optimize_method='geometric'
)

# ========== 第5组：梯度提升 ==========
models["Dual_Gradient_Boosting"] = DualOptimizerClassifier(
    base_model=GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
    alpha=0.5,
    optimize_method='custom'
)

models["Dual_AdaBoost"] = DualOptimizerClassifier(
    base_model=AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    ),
    alpha=0.5,
    optimize_method='custom'
)

# ========== 第6组：Bagging ==========
models["Dual_Bagging_DT"] = DualOptimizerClassifier(
    base_model=BaggingClassifier(
        base_estimator=DecisionTreeClassifier(class_weight='balanced'),
        n_estimators=50,
        random_state=42
    ),
    alpha=0.5,
    optimize_method='f2_pr'
)

# ========== 第7组：神经网络 ==========
models["Dual_MLP_Deep"] = DualOptimizerClassifier(
    base_model=MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    ),
    alpha=0.5,
    optimize_method='f2_pr'
)

# ========== 第8组：贝叶斯 ==========
# GaussianNB不支持class_weight，保持原样
models["Gaussian_Naive_Bayes"] = GaussianNB()

# ========== 第9组：判别分析 ==========
models["Linear_Discriminant"] = LinearDiscriminantAnalysis()
models["Quadratic_Discriminant"] = QuadraticDiscriminantAnalysis()

# 添加双提升优化版本
models["Dual_Linear_Discriminant"] = DualOptimizerClassifier(
    base_model=LinearDiscriminantAnalysis(solver='svd'),
    alpha=0.5,
    optimize_method='f2_pr'
)

models["Dual_Quadratic_Discriminant"] = DualOptimizerClassifier(
    base_model=QuadraticDiscriminantAnalysis(reg_param=0.01),  # 添加正则化防止奇异矩阵
    alpha=0.5,
    optimize_method='f2_pr'
)

# ========== 第10组：K近邻 ==========
models["Dual_KNN_k5"] = DualOptimizerClassifier(
    base_model=KNeighborsClassifier(n_neighbors=5),
    alpha=0.5,
    optimize_method='geometric'
)

models["Dual_KNN_k10"] = DualOptimizerClassifier(
    base_model=KNeighborsClassifier(n_neighbors=10),
    alpha=0.5,
    optimize_method='geometric'
)

# ========== 第11组：XGBoost ==========
if XGB_AVAILABLE:
    models["Dual_XGBoost"] = DualOptimizerClassifier(
        base_model=XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        alpha=0.45,
        optimize_method='f2_pr'
    )

# ========== 第12组：堆叠集成 ==========
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

if XGB_AVAILABLE:
    base_estimators.insert(1, (
        'xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')))

stacking_model = StackingClassifier(
    estimators=base_estimators[:3],
    final_estimator=LogisticRegression(),
    cv=3
)

models["Dual_Stacking_Ensemble"] = DualOptimizerClassifier(
    base_model=stacking_model,
    alpha=0.5,
    optimize_method='f2_pr'
)

# ========== 第13组：传统模型（作为对比） ==========
models["RF_Baseline"] = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

models["LR_Baseline"] = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# ========== 第14组：传统评分模型（如果存在） ==========
if traditional_score_cols:
    print("\n添加传统评分模型作为基准...")
    for score_type, col_name in traditional_score_cols:
        models[f"Traditional_{score_type}"] = TraditionalScoreClassifier(
            score_type=score_type,
            optimal_threshold=0.5
        )
    print(f"  添加了 {len(traditional_score_cols)} 个传统评分模型")

print(f"\n总共 {len(models)} 个模型将被测试")
print("模型列表:", list(models.keys()))


# ========== 综合评估函数 ==========
def evaluate_comprehensive_metrics(y_true, y_pred, y_pred_proba=None, model_name=None):
    """评估所有指标"""
    metrics = {}

    # 基础分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # F-beta系列
    metrics['f0.5'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    metrics['f1.5'] = fbeta_score(y_true, y_pred, beta=1.5, zero_division=0)

    # 几何指标
    if metrics['precision'] > 0 and metrics['recall'] > 0:
        metrics['gmean'] = np.sqrt(metrics['precision'] * metrics['recall'])
    else:
        metrics['gmean'] = 0

    # 平衡分数（roc4_2中的指标）
    metrics['balanced_score'] = 0.4 * metrics['precision'] + 0.4 * metrics['recall'] + 0.2 * metrics['f1']

    # PR AUC
    if y_pred_proba is not None and len(np.unique(y_pred_proba)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        if len(recall_curve) > 1 and len(precision_curve) > 1:
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        else:
            metrics['pr_auc'] = np.mean(y_true)
    else:
        metrics['pr_auc'] = np.mean(y_true)

    # ROC AUC
    if y_pred_proba is not None and len(np.unique(y_pred_proba)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    else:
        metrics['roc_auc'] = 0.5

    # 混淆矩阵指标
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp

        # 敏感性、特异性等
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值

        # 额外指标
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['diagnostic_odds_ratio'] = (tp / fp) / (fn / tn) if fp > 0 and fn > 0 and tn > 0 else 0
    else:
        metrics['sensitivity'] = metrics['specificity'] = metrics['ppv'] = metrics['npv'] = 0
        metrics['false_positive_rate'] = metrics['false_negative_rate'] = 0
        metrics['diagnostic_odds_ratio'] = 0

    # 马修斯相关系数
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics['mcc'] = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    # 平衡准确率
    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2

    # Youden指数
    metrics['youden_index'] = metrics['sensitivity'] + metrics['specificity'] - 1

    return metrics


# ========== 设置交叉验证 ==========
n_folds = 30  # 使用30折，如roc3.py中一样
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ========== 存储结果 ==========
results = {}
all_fold_results = []  # 存储每一折的详细结果
model_summaries = []  # 存储模型汇总结果
all_roc_data = {}  # 存储ROC曲线数据
all_pr_data = {}  # 存储PR曲线数据

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_combined_traditional_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ========== 训练和评估循环 ==========
print(f"\n{'=' * 80}")
print("开始训练和评估...")
print("=" * 80)

for model_idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"\n{'=' * 80}")
    print(f"训练模型 [{model_idx}/{len(models)}]: {model_name}")
    print('=' * 80)

    # 存储每一折的指标
    fold_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'f0.5': [], 'f2': [], 'f1.5': [], 'gmean': [],
        'balanced_score': [], 'pr_auc': [], 'roc_auc': [],
        'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': [],
        'mcc': [], 'balanced_accuracy': [], 'youden_index': [],
        'optimal_threshold': []
    }

    # 存储ROC和PR曲线数据
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    fold = 0
    successful_folds = 0

    for train_idx, test_idx in skf.split(X_original, y):
        fold += 1

        # 根据模型类型准备数据
        if model_name.startswith('Traditional_'):
            # 传统评分模型：使用完整的原始数据
            X_train_full = X_with_scores_full.iloc[train_idx]
            X_test_full = X_with_scores_full.iloc[test_idx]
        else:
            # 机器学习模型：使用标准化后的特征数据
            # 在每一折中单独进行标准化，避免数据泄露
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_original[train_idx])
            X_test = fold_scaler.transform(X_original[test_idx])

        y_train, y_test = y[train_idx], y[test_idx]

        try:
            # 训练模型
            if model_name.startswith('Traditional_'):
                # 传统评分模型
                model.fit(X_train_full, y_train)
                optimal_threshold = 0.5  # 传统评分使用固定阈值或从结果中获取
            else:
                model.fit(X_train, y_train)

                # 获取最优阈值（对于双优化器）
                if isinstance(model, DualOptimizerClassifier):
                    optimal_threshold = model.optimal_threshold
                else:
                    optimal_threshold = 0.5

            # 预测
            if model_name.startswith('Traditional_'):
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_full)[:, 1]
                    y_pred = model.predict(X_test_full)
                else:
                    y_pred = model.predict(X_test_full)
                    y_proba = y_pred.astype(float)
            else:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    y_proba = expit(decision_scores)
                    y_pred = (y_proba >= optimal_threshold).astype(int)
                else:
                    y_pred = model.predict(X_test)
                    y_proba = y_pred.astype(float)

            # 评估指标
            metrics = evaluate_comprehensive_metrics(y_test, y_pred, y_proba, model_name)

            # 存储指标
            for key in fold_metrics.keys():
                if key in metrics:
                    fold_metrics[key].append(metrics[key])

            fold_metrics['optimal_threshold'].append(optimal_threshold)

            # 存储ROC曲线数据
            if 'fpr' in metrics and 'tpr' in metrics:
                tprs.append(np.interp(mean_fpr, metrics['fpr'], metrics['tpr']))
                tprs[-1][0] = 0.0
                aucs.append(metrics['roc_auc'])

            # 存储PR曲线数据
            if y_proba is not None and len(np.unique(y_proba)) > 1:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
                if len(recall_vals) > 1 and len(precision_vals) > 1:
                    precision_interp = np.interp(mean_recall, recall_vals[::-1], precision_vals[::-1])
                    precisions.append(precision_interp)
                    pr_aucs.append(metrics['pr_auc'])

            # 存储详细fold结果
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'F0.5_Score': metrics['f0.5'],
                'F2_Score': metrics['f2'],
                'F1.5_Score': metrics['f1.5'],
                'G_Mean': metrics['gmean'],
                'Balanced_Score': metrics['balanced_score'],
                'ROC_AUC': metrics['roc_auc'],
                'PR_AUC': metrics['pr_auc'],
                'Sensitivity_TPR': metrics['sensitivity'],
                'Specificity_TNR': metrics['specificity'],
                'PPV': metrics['ppv'],
                'NPV': metrics['npv'],
                'MCC': metrics['mcc'],
                'Balanced_Accuracy': metrics['balanced_accuracy'],
                'Youden_Index': metrics['youden_index'],
                'False_Positive_Rate': metrics.get('false_positive_rate', 0),
                'False_Negative_Rate': metrics.get('false_negative_rate', 0),
                'Diagnostic_OR': metrics.get('diagnostic_odds_ratio', 0),
                'Optimal_Threshold': optimal_threshold,
                'TP': metrics.get('tp', 0),
                'FP': metrics.get('fp', 0),
                'TN': metrics.get('tn', 0),
                'FN': metrics.get('fn', 0),
                'Train_Samples': len(train_idx),
                'Test_Samples': len(test_idx),
                'Test_Positive_Rate': np.mean(y_test),
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_fold_results.append(fold_result)

            successful_folds += 1

            # 每5个fold打印一次进度
            if fold % 5 == 0:
                print(f"  已完成 {fold}/{n_folds} folds...")

        except Exception as e:
            print(f"  Fold {fold} 失败: {str(e)[:100]}...")

            # 记录失败的fold
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Error': str(e)[:200],
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_fold_results.append(fold_result)
            continue

    if successful_folds == 0:
        print(f"模型 {model_name} 所有fold失败，跳过")
        continue

    # 计算平均指标和标准差
    mean_metrics = {}
    std_metrics = {}

    for key in fold_metrics.keys():
        if fold_metrics[key]:
            mean_metrics[f'mean_{key}'] = np.nanmean(fold_metrics[key])
            std_metrics[f'std_{key}'] = np.nanstd(fold_metrics[key])

    # 合并结果
    final_metrics = {**fold_metrics, **mean_metrics, **std_metrics}
    results[model_name] = final_metrics

    # 存储曲线数据
    if len(tprs) > 0:
        all_roc_data[model_name] = {
            'mean_fpr': mean_fpr,
            'mean_tpr': np.nanmean(tprs, axis=0),
            'std_tpr': np.nanstd(tprs, axis=0),
            'mean_auc': np.nanmean(aucs),
            'std_auc': np.nanstd(aucs)
        }

    if len(precisions) > 0:
        all_pr_data[model_name] = {
            'mean_recall': mean_recall,
            'mean_precision': np.nanmean(precisions, axis=0),
            'std_precision': np.nanstd(precisions, axis=0),
            'mean_pr_auc': np.nanmean(pr_aucs),
            'std_pr_auc': np.nanstd(pr_aucs)
        }

    # 创建模型汇总
    summary = {
        'Model': model_name,
        'Total_Folds': n_folds,
        'Successful_Folds': successful_folds,
        'Success_Rate': successful_folds / n_folds * 100,
        'Mean_Precision': mean_metrics.get('mean_precision', np.nan),
        'Std_Precision': std_metrics.get('std_precision', np.nan),
        'Mean_Recall': mean_metrics.get('mean_recall', np.nan),
        'Std_Recall': std_metrics.get('std_recall', np.nan),
        'Mean_F1': mean_metrics.get('mean_f1', np.nan),
        'Std_F1': std_metrics.get('std_f1', np.nan),
        'Mean_F2': mean_metrics.get('mean_f2', np.nan),
        'Mean_F0.5': mean_metrics.get('mean_f0.5', np.nan),
        'Mean_Balanced_Score': mean_metrics.get('mean_balanced_score', np.nan),
        'Std_Balanced_Score': std_metrics.get('std_balanced_score', np.nan),
        'Mean_PR_AUC': mean_metrics.get('mean_pr_auc', np.nan),
        'Std_PR_AUC': std_metrics.get('std_pr_auc', np.nan),
        'Mean_ROC_AUC': mean_metrics.get('mean_roc_auc', np.nan),
        'Std_ROC_AUC': std_metrics.get('std_roc_auc', np.nan),
        'Mean_Sensitivity': mean_metrics.get('mean_sensitivity', np.nan),
        'Mean_Specificity': mean_metrics.get('mean_specificity', np.nan),
        'Mean_MCC': mean_metrics.get('mean_mcc', np.nan),
        'Mean_Balanced_Accuracy': mean_metrics.get('mean_balanced_accuracy', np.nan),
        'Mean_Youden_Index': mean_metrics.get('mean_youden_index', np.nan),
        'Mean_Optimal_Threshold': mean_metrics.get('mean_optimal_threshold', np.nan),
        'Evaluation_Date': datetime.now().strftime("%Y-%m-%d")
    }
    model_summaries.append(summary)

    # 打印当前模型性能
    print(f"\n{model_name} 性能汇总 ({successful_folds}/{n_folds} folds成功):")
    print(
        f"  Precision: {mean_metrics.get('mean_precision', np.nan):.4f} ± {std_metrics.get('std_precision', np.nan):.4f}")
    print(f"  Recall:    {mean_metrics.get('mean_recall', np.nan):.4f} ± {std_metrics.get('std_recall', np.nan):.4f}")
    print(f"  F1:        {mean_metrics.get('mean_f1', np.nan):.4f} ± {std_metrics.get('std_f1', np.nan):.4f}")
    print(f"  Balanced:  {mean_metrics.get('mean_balanced_score', np.nan):.4f}")
    print(f"  PR AUC:    {mean_metrics.get('mean_pr_auc', np.nan):.4f} ± {std_metrics.get('std_pr_auc', np.nan):.4f}")
    print(f"  ROC AUC:   {mean_metrics.get('mean_roc_auc', np.nan):.4f} ± {std_metrics.get('std_roc_auc', np.nan):.4f}")

# ========== 保存结果到CSV文件 ==========
print(f"\n{'=' * 80}")
print("正在保存结果到CSV文件...")
print("=" * 80)

# 1. 保存每一折的详细结果
if all_fold_results:
    fold_df = pd.DataFrame(all_fold_results)
    fold_csv_path = os.path.join(results_dir, 'fold_results_detailed.csv')
    fold_df.to_csv(fold_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存每一折详细结果到: {fold_csv_path}")
    print(f"  共 {len(fold_df)} 条记录，{len(fold_df.columns)} 个指标")

# 2. 保存模型汇总结果
if model_summaries:
    summary_df = pd.DataFrame(model_summaries)
    summary_csv_path = os.path.join(results_dir, 'model_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存模型汇总结果到: {summary_csv_path}")
    print(f"  共 {len(summary_df)} 个模型，{len(summary_df.columns)} 个指标")

# 3. 创建详细排名表
if model_summaries and 'summary_df' in locals():
    # 按不同指标排名
    summary_df['PR_AUC_Rank'] = summary_df['Mean_PR_AUC'].rank(ascending=False)
    summary_df['F1_Rank'] = summary_df['Mean_F1'].rank(ascending=False)
    summary_df['Balanced_Score_Rank'] = summary_df['Mean_Balanced_Score'].rank(ascending=False)
    summary_df['Youden_Rank'] = summary_df['Mean_Youden_Index'].rank(ascending=False)

    # 综合排名
    summary_df['Overall_Rank'] = (
            summary_df['PR_AUC_Rank'] * 0.3 +
            summary_df['F1_Rank'] * 0.25 +
            summary_df['Balanced_Score_Rank'] * 0.25 +
            summary_df['Youden_Rank'] * 0.2
    )

    # 按综合排名排序
    summary_df = summary_df.sort_values('Overall_Rank')

    # 保存排名结果
    rank_csv_path = os.path.join(results_dir, 'model_ranking.csv')
    summary_df.to_csv(rank_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存模型排名到: {rank_csv_path}")

    # 显示前10名
    print(f"\n{'=' * 80}")
    print("Top 10 模型 (按综合排名):")
    print('=' * 80)

    display_cols = ['Model', 'Mean_PR_AUC', 'Mean_F1', 'Mean_Balanced_Score',
                    'Mean_Recall', 'Mean_Precision', 'Mean_ROC_AUC', 'Overall_Rank']
    print(summary_df[display_cols].head(10).to_string(index=False))

# ========== 传统评分与机器学习模型对比分析 ==========
if traditional_score_cols and len(results) > 0:
    print(f"\n{'=' * 80}")
    print("传统评分与机器学习模型对比分析")
    print("=" * 80)

    # 创建对比数据
    comparison_data = []

    # 添加传统评分结果
    for score_type in traditional_score_results:
        model_name = f"Traditional_{score_type}"
        if model_name in results:
            model_results = results[model_name]
            comparison_data.append({
                'Model_Type': 'Traditional',
                'Model_Name': model_name,
                'Mean_PR_AUC': model_results.get('mean_pr_auc', np.nan),
                'Std_PR_AUC': model_results.get('std_pr_auc', np.nan),
                'Mean_ROC_AUC': model_results.get('mean_roc_auc', np.nan),
                'Std_ROC_AUC': model_results.get('std_roc_auc', np.nan),
                'Mean_Precision': model_results.get('mean_precision', np.nan),
                'Std_Precision': model_results.get('std_precision', np.nan),
                'Mean_Recall': model_results.get('mean_recall', np.nan),
                'Std_Recall': model_results.get('std_recall', np.nan),
                'Mean_F1_Score': model_results.get('mean_f1', np.nan),
                'Std_F1_Score': model_results.get('std_f1', np.nan),
                'Mean_Balanced_Score': model_results.get('mean_balanced_score', np.nan),
                'Std_Balanced_Score': model_results.get('std_balanced_score', np.nan),
                'Mean_Optimal_Threshold': model_results.get('mean_optimal_threshold', np.nan),
                'Std_Optimal_Threshold': model_results.get('std_optimal_threshold', np.nan)
            })

    # 添加机器学习模型（前5名）
    ml_models = [m for m in results.keys() if not m.startswith('Traditional_')]
    ml_models_sorted = sorted(ml_models,
                              key=lambda x: results[x].get('mean_pr_auc', 0),
                              reverse=True)[:5]

    for model_name in ml_models_sorted:
        model_results = results[model_name]
        comparison_data.append({
            'Model_Type': 'Machine_Learning',
            'Model_Name': model_name,
            'Mean_PR_AUC': model_results.get('mean_pr_auc', np.nan),
            'Std_PR_AUC': model_results.get('std_pr_auc', np.nan),
            'Mean_ROC_AUC': model_results.get('mean_roc_auc', np.nan),
            'Std_ROC_AUC': model_results.get('std_roc_auc', np.nan),
            'Mean_Precision': model_results.get('mean_precision', np.nan),
            'Std_Precision': model_results.get('std_precision', np.nan),
            'Mean_Recall': model_results.get('mean_recall', np.nan),
            'Std_Recall': model_results.get('std_recall', np.nan),
            'Mean_F1_Score': model_results.get('mean_f1', np.nan),
            'Std_F1_Score': model_results.get('std_f1', np.nan),
            'Mean_Balanced_Score': model_results.get('mean_balanced_score', np.nan),
            'Std_Balanced_Score': model_results.get('std_balanced_score', np.nan),
            'Mean_Optimal_Threshold': model_results.get('mean_optimal_threshold', np.nan),
            'Std_Optimal_Threshold': model_results.get('std_optimal_threshold', np.nan)
        })

    # 创建对比DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # 保存对比数据
    comparison_csv_path = os.path.join(results_dir, 'traditional_vs_ml_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存对比数据到: {comparison_csv_path}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PR AUC对比条形图
    ax1 = axes[0, 0]
    comparison_sorted = comparison_df.sort_values('Mean_PR_AUC', ascending=False)
    colors = ['#1f77b4' if t == 'Traditional' else '#2ca02c' for t in comparison_sorted['Model_Type']]
    y_pos = np.arange(len(comparison_sorted))
    bars = ax1.barh(y_pos, comparison_sorted['Mean_PR_AUC'],
                    xerr=comparison_sorted['Std_PR_AUC'],
                    color=colors, alpha=0.8, ecolor='black', capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comparison_sorted['Model_Name'], fontsize=9)
    ax1.set_xlabel('PR AUC (均值±标准差)', fontsize=11)
    ax1.set_title('传统评分 vs 机器学习模型 PR AUC 对比', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#1f77b4', label='传统评分'),
        Patch(facecolor='#2ca02c', label='机器学习模型')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # 2. 精确率-召回率散点图
    ax2 = axes[0, 1]
    traditional_df = comparison_df[comparison_df['Model_Type'] == 'Traditional']
    ml_df = comparison_df[comparison_df['Model_Type'] == 'Machine_Learning']
    ax2.scatter(traditional_df['Mean_Recall'], traditional_df['Mean_Precision'],
                s=150, c='blue', marker='s', alpha=0.8, label='传统评分', edgecolors='black')
    ax2.scatter(ml_df['Mean_Recall'], ml_df['Mean_Precision'],
                s=150, c='green', marker='o', alpha=0.8, label='机器学习', edgecolors='black')
    ax2.set_xlabel('召回率 (Recall)', fontsize=11)
    ax2.set_ylabel('精确率 (Precision)', fontsize=11)
    ax2.set_title('精确率-召回率对比', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    # 3. 性能提升百分比
    ax3 = axes[1, 0]
    if not traditional_df.empty:
        best_traditional_idx = traditional_df['Mean_PR_AUC'].idxmax()
        best_traditional = traditional_df.loc[best_traditional_idx]
        improvements = []
        model_names = []

        for idx, row in ml_df.iterrows():
            improvement = ((row['Mean_PR_AUC'] - best_traditional['Mean_PR_AUC']) /
                           best_traditional['Mean_PR_AUC']) * 100
            improvements.append(improvement)
            model_names.append(row['Model_Name'])

        if improvements:
            x_pos = np.arange(len(ml_df))
            bars = ax3.bar(x_pos, improvements,
                           color=['green' if imp > 0 else 'red' for imp in improvements])
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('PR AUC提升百分比 (%)', fontsize=11)
            ax3.set_title(f'机器学习模型相对于最佳传统评分({best_traditional["Model_Name"]})的性能提升',
                          fontsize=12, fontweight='bold')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2.,
                         height + (1 if height >= 0 else -3),
                         f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=9, fontweight='bold')

    # 4. 阈值分布对比
    ax4 = axes[1, 1]
    thresholds_traditional = traditional_df['Mean_Optimal_Threshold'].values
    thresholds_ml = ml_df['Mean_Optimal_Threshold'].values
    data_to_plot = [thresholds_traditional, thresholds_ml]
    bp = ax4.boxplot(data_to_plot, labels=['传统评分', '机器学习'], patch_artist=True)
    colors_box = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    ax4.set_ylabel('最优阈值', fontsize=11)
    ax4.set_title('最优阈值分布对比', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, 'traditional_vs_ml_comparison_plot.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存对比分析图到: {comparison_plot_path}")

    # 生成对比报告
    generate_comparison_report(comparison_df, results_dir, y)


# ========== 生成对比报告 ==========
def generate_comparison_report(comparison_df, results_dir, y):
    """生成详细的对比报告"""

    report_content = []
    report_content.append("=" * 80)
    report_content.append("传统评分 vs 机器学习模型对比分析报告")
    report_content.append("=" * 80)
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"数据集信息: 总样本数={len(y)}, 正类比例={np.mean(y) * 100:.2f}%")
    report_content.append("")

    # 1. 最佳模型识别
    report_content.append("1. 最佳模型识别")
    report_content.append("-" * 40)

    # 最佳传统评分
    traditional_df = comparison_df[comparison_df['Model_Type'] == 'Traditional']
    if not traditional_df.empty:
        best_traditional_idx = traditional_df['Mean_PR_AUC'].idxmax()
        best_traditional = traditional_df.loc[best_traditional_idx]
        report_content.append(f"   最佳传统评分: {best_traditional['Model_Name']}")
        report_content.append(
            f"      PR AUC: {best_traditional['Mean_PR_AUC']:.4f} ± {best_traditional['Std_PR_AUC']:.4f}")
        report_content.append(
            f"      精确率: {best_traditional['Mean_Precision']:.4f} ± {best_traditional['Std_Precision']:.4f}")
        report_content.append(
            f"      召回率: {best_traditional['Mean_Recall']:.4f} ± {best_traditional['Std_Recall']:.4f}")
        report_content.append(
            f"      平衡分数: {best_traditional['Mean_Balanced_Score']:.4f} ± {best_traditional['Std_Balanced_Score']:.4f}")

    # 最佳机器学习模型
    ml_df = comparison_df[comparison_df['Model_Type'] == 'Machine_Learning']
    if not ml_df.empty:
        best_ml_idx = ml_df['Mean_PR_AUC'].idxmax()
        best_ml = ml_df.loc[best_ml_idx]
        report_content.append(f"   最佳机器学习模型: {best_ml['Model_Name']}")
        report_content.append(f"      PR AUC: {best_ml['Mean_PR_AUC']:.4f} ± {best_ml['Std_PR_AUC']:.4f}")
        report_content.append(f"      精确率: {best_ml['Mean_Precision']:.4f} ± {best_ml['Std_Precision']:.4f}")
        report_content.append(f"      召回率: {best_ml['Mean_Recall']:.4f} ± {best_ml['Std_Recall']:.4f}")
        report_content.append(f"      平衡分数: {best_ml['Mean_Balanced_Score']:.4f} ± {best_ml['Std_Balanced_Score']:.4f}")

    # 2. 性能对比分析
    report_content.append("")
    report_content.append("2. 性能对比分析")
    report_content.append("-" * 40)

    if not traditional_df.empty and not ml_df.empty:
        # 计算平均提升
        avg_prauc_traditional = traditional_df['Mean_PR_AUC'].mean()
        avg_prauc_ml = ml_df['Mean_PR_AUC'].mean()
        prauc_improvement = ((avg_prauc_ml - avg_prauc_traditional) / avg_prauc_traditional) * 100

        report_content.append(f"   传统评分平均PR AUC: {avg_prauc_traditional:.4f}")
        report_content.append(f"   机器学习模型平均PR AUC: {avg_prauc_ml:.4f}")
        report_content.append(f"   机器学习模型平均提升: {prauc_improvement:.2f}%")

        # 最佳对比
        if 'best_traditional' in locals() and 'best_ml' in locals():
            best_improvement = ((best_ml['Mean_PR_AUC'] - best_traditional['Mean_PR_AUC']) /
                                best_traditional['Mean_PR_AUC']) * 100
            report_content.append(f"   最佳机器学习模型相对于最佳传统评分的提升: {best_improvement:.2f}%")

    # 3. 统计显著性检验
    report_content.append("")
    report_content.append("3. 统计显著性检验")
    report_content.append("-" * 40)

    # 简单的均值比较
    if not traditional_df.empty and not ml_df.empty:
        traditional_prauc = traditional_df['Mean_PR_AUC'].values
        ml_prauc = ml_df['Mean_PR_AUC'].values

        if len(traditional_prauc) > 1 and len(ml_prauc) > 1:
            # 使用t检验比较均值
            t_stat, p_value = stats.ttest_ind(traditional_prauc, ml_prauc)
            report_content.append(f"   独立样本t检验结果:")
            report_content.append(f"     t统计量: {t_stat:.4f}")
            report_content.append(f"     p值: {p_value:.4f}")

            if p_value < 0.05:
                report_content.append(f"     结论: 在0.05显著性水平下，机器学习模型与传统评分有显著差异")
            else:
                report_content.append(f"     结论: 在0.05显著性水平下，机器学习模型与传统评分无显著差异")

    # 4. 临床意义和建议
    report_content.append("")
    report_content.append("4. 临床意义和建议")
    report_content.append("-" * 40)

    if 'best_improvement' in locals():
        if best_improvement > 10:
            report_content.append("   ✅ 临床意义显著:")
            report_content.append("     机器学习模型性能显著优于传统评分（提升>10%）")
            report_content.append("     建议在临床实践中考虑采用机器学习模型")
            report_content.append("     可显著提高ARDS预测准确性")
        elif best_improvement > 5:
            report_content.append("   ⚠ 有一定临床价值:")
            report_content.append("     机器学习模型性能有一定提升（5-10%）")
            report_content.append("     可作为传统评分的补充工具")
            report_content.append("     在特定场景下可能更有价值")
        elif best_improvement > 0:
            report_content.append("   ℹ 提升有限:")
            report_content.append("     机器学习模型性能提升有限（<5%）")
            report_content.append("     传统评分因其简单、透明性仍具有优势")
            report_content.append("     可考虑进一步优化机器学习模型")
        else:
            report_content.append("   ⚠ 传统评分仍具优势:")
            report_content.append("     传统评分性能更优或相当")
            report_content.append("     建议继续使用传统评分")
            report_content.append("     机器学习模型可能需要进一步调优")

    # 5. 实施建议
    report_content.append("")
    report_content.append("5. 实施建议")
    report_content.append("-" * 40)

    if 'best_ml' in locals():
        optimal_threshold = best_ml['Mean_Optimal_Threshold']
        report_content.append(f"   最佳机器学习模型实施建议:")
        report_content.append(f"     推荐阈值: {optimal_threshold:.3f}")
        report_content.append(f"     在此阈值下:")
        report_content.append(f"       - 精确率: {best_ml['Mean_Precision']:.3f}")
        report_content.append(f"       - 召回率: {best_ml['Mean_Recall']:.3f}")
        report_content.append(f"       - 平衡分数: {best_ml['Mean_Balanced_Score']:.3f}")

        if best_ml['Mean_Precision'] > best_ml['Mean_Recall']:
            report_content.append(f"      特点: 偏向高精确率（假阳性率低）")
            report_content.append(f"      适用场景: 资源有限，希望减少假阳性的情况")
        else:
            report_content.append(f"      特点: 偏向高召回率（漏诊率低）")
            report_content.append(f"      适用场景: ARDS风险高，希望尽可能发现所有病例")

    report_content.append("")
    report_content.append("=" * 80)

    # 保存报告
    report_path = os.path.join(results_dir, 'traditional_vs_ml_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))

    # 打印报告摘要
    print("\n" + "\n".join(report_content[:50]))  # 打印前50行
    print(f"\n... (完整报告已保存到文件)")
    print(f"已保存详细对比报告到: {report_path}")


# ========== 可视化 ==========
if results:
    print(f"\n{'=' * 80}")
    print("生成可视化图表...")
    print("=" * 80)

    # 1. 性能对比条形图
    if 'summary_df' in locals():
        top_models = summary_df.head(8)['Model'].values

        plt.figure(figsize=(14, 8))

        metrics_to_plot = ['Mean_Precision', 'Mean_Recall', 'Mean_F1', 'Mean_PR_AUC']
        x = np.arange(len(top_models))
        width = 0.15

        for i, metric in enumerate(metrics_to_plot):
            values = summary_df.loc[summary_df['Model'].isin(top_models), metric].values
            if len(values) == len(top_models):
                plt.bar(x + i * width - width * 1.5, values, width, label=metric.replace('Mean_', ''))

        plt.xlabel('模型')
        plt.ylabel('分数')
        plt.title('Top 8 模型指标对比')
        plt.xticks(x, [m[:15] for m in top_models], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        bar_path = os.path.join(results_dir, 'top_models_comparison.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"已保存条形图: {bar_path}")

    # 2. Precision-Recall散点图
    plt.figure(figsize=(12, 10))

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for idx, (model_name, metrics) in enumerate(results.items()):
        precision = metrics.get('mean_precision', 0)
        recall = metrics.get('mean_recall', 0)
        f1 = metrics.get('mean_f1', 0)

        plt.scatter(recall, precision, s=f1 * 200, color=colors[idx],
                    alpha=0.7, edgecolors='k')

        # 标注模型名（缩短）
        short_name = model_name.replace('Dual_', '').replace('_', ' ')[:15]
        plt.annotate(short_name, (recall, precision), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    plt.xlabel('Recall (敏感性)', fontsize=12)
    plt.ylabel('Precision (精确率)', fontsize=12)
    plt.title('Precision vs Recall (气泡大小=F1分数)', fontsize=14)
    plt.grid(True, alpha=0.3)

    scatter_path = os.path.join(results_dir, 'precision_recall_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存散点图: {scatter_path}")

    # 3. ROC曲线对比
    if len(all_roc_data) > 0:
        # 选择PR AUC最高的几个模型进行ROC曲线绘制
        top_models_roc = sorted(results.items(),
                                key=lambda x: x[1].get('mean_pr_auc', 0),
                                reverse=True)[:6]

        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_models_roc)))

        for idx, (model_name, _) in enumerate(top_models_roc):
            roc_info = all_roc_data.get(model_name)
            if roc_info is not None:
                mean_tpr = roc_info['mean_tpr']
                mean_fpr = roc_info['mean_fpr']
                mean_auc = roc_info['mean_auc']
                std_auc = roc_info['std_auc']

                plt.plot(mean_fpr, mean_tpr, color=colors[idx],
                         label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                         lw=2, alpha=0.8)

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves Comparison - Dual Optimized Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_plot_path = os.path.join(results_dir, 'ROC_Curves_Comparison.png')
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"已保存ROC曲线图: {roc_plot_path}")

    # 4. PR曲线对比
    if len(all_pr_data) > 0:
        # 选择PR AUC最高的几个模型
        top_models_pr = sorted(results.items(),
                               key=lambda x: x[1].get('mean_pr_auc', 0),
                               reverse=True)[:6]

        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_models_pr)))

        pos_rate = np.mean(y)

        for idx, (model_name, _) in enumerate(top_models_pr):
            pr_info = all_pr_data.get(model_name)
            if pr_info is not None:
                mean_precision = pr_info['mean_precision']
                mean_recall = pr_info['mean_recall']
                mean_pr_auc = pr_info['mean_pr_auc']
                std_pr_auc = pr_info['std_pr_auc']

                plt.plot(mean_recall, mean_precision, color=colors[idx],
                         label=f'{model_name} (AP = {mean_pr_auc:.3f} ± {std_pr_auc:.3f})',
                         lw=2, alpha=0.8)

        plt.axhline(y=pos_rate, color='gray', linestyle='--', lw=1, alpha=0.5,
                    label=f'Random (AP = {pos_rate:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
        plt.title('Precision-Recall Curves - Top Models', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_plot_path = os.path.join(results_dir, 'PR_Curves_Comparison.png')
        plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"已保存PR曲线图: {pr_plot_path}")

# ========== 生成详细分析报告 ==========
print(f"\n{'=' * 80}")
print("详细分析报告")
print("=" * 80)

if model_summaries and 'summary_df' in locals():
    best_model = summary_df.iloc[0]
    best_name = best_model['Model']

    print(f"1. 最佳模型: {best_name}")
    print(f"   - PR AUC: {best_model['Mean_PR_AUC']:.4f}")
    print(f"   - F1 Score: {best_model['Mean_F1']:.4f}")
    print(f"   - Balanced Score: {best_model['Mean_Balanced_Score']:.4f}")
    print(f"   - 召回率(Recall): {best_model['Mean_Recall']:.4f}")
    print(f"   - 精确率(Precision): {best_model['Mean_Precision']:.4f}")
    print(f"   - Youden指数: {best_model['Mean_Youden_Index']:.4f}")

    print(f"\n2. 数据集统计:")
    print(f"   - 总样本数: {len(y)}")
    print(f"   - 正类比例: {np.mean(y):.4f} ({sum(y)}/{len(y)})")
    print(f"   - 随机模型PR AUC基线: {np.mean(y):.4f}")

    print(f"\n3. 双提升优化效果分析:")
    dual_models = [m for m in results.keys() if m.startswith('Dual_')]
    baseline_models = [m for m in results.keys() if
                       not m.startswith('Dual_') and not m.startswith('Gaussian') and not m.startswith(
                           'Linear') and not m.startswith('Quadratic') and not m.startswith('Traditional_')]

    if dual_models and baseline_models:
        dual_pr_auc = np.mean([results[m].get('mean_pr_auc', 0) for m in dual_models])
        baseline_pr_auc = np.mean([results[m].get('mean_pr_auc', 0) for m in baseline_models])

        print(f"   - 双提升模型平均PR AUC: {dual_pr_auc:.4f}")
        print(f"   - 传统模型平均PR AUC: {baseline_pr_auc:.4f}")
        improvement = ((dual_pr_auc - baseline_pr_auc) / baseline_pr_auc * 100) if baseline_pr_auc > 0 else 0
        print(f"   - 平均提升: {improvement:.1f}%")

    # 传统评分对比分析
    if traditional_score_cols:
        print(f"\n4. 传统评分对比:")
        traditional_models = [m for m in results.keys() if m.startswith('Traditional_')]
        if traditional_models:
            traditional_pr_auc = np.mean([results[m].get('mean_pr_auc', 0) for m in traditional_models])
            print(f"   - 传统评分平均PR AUC: {traditional_pr_auc:.4f}")
            if 'best_model' in locals() and best_model['Mean_PR_AUC'] > 0:
                improvement_vs_traditional = (
                        (best_model['Mean_PR_AUC'] - traditional_pr_auc) / traditional_pr_auc * 100)
                print(f"   - 最佳模型相对于传统评分提升: {improvement_vs_traditional:.1f}%")

    print(f"\n5. 临床推荐:")
    if best_model['Mean_Recall'] > 0.8 and best_model['Mean_Precision'] > 0.7:
        print("   ✓ 模型具有高召回率和高精确率 - 适合临床诊断")
    elif best_model['Mean_Recall'] > 0.8:
        print("   ⚠ 模型具有高召回率但精确率一般 - 适合筛查（漏诊风险低）")
    elif best_model['Mean_Precision'] > 0.7:
        print("   ⚠ 模型具有高精确率但召回率一般 - 适合确诊（误诊风险低）")
    else:
        print("   ✗ 模型性能需进一步提升")

print(f"\n{'=' * 80}")
print("所有分析完成!")
print(f"结果保存在: {results_dir}")
print("=" * 80)

# 显示文件列表
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f"\n生成的文件:")
    for file in sorted(files):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  - {file} ({size_kb:.1f} KB)")