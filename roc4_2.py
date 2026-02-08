
# 同时优化Precision和recall最大值的版本

# 综合版本：集成roc3.py的所有模型，应用roc4_2.py的双提升优化策略

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


# ========== 数据加载和预处理 ==========
print("=" * 80)
print("加载和预处理数据...")
print("=" * 80)

data = pd.read_csv('SAP6_ARDS.csv')
np.set_printoptions(suppress=True)
y = data["ARDS"].values
X = data.iloc[:, 0:32].values


print(f"数据集信息:")
print(f"  总样本数: {len(y)}")
print(f"  特征维度: {X.shape[1]}")
print(f"  正类(SAP)数量: {sum(y)} ({np.mean(y) * 100:.2f}%)")
print(f"  负类数量: {len(y) - sum(y)} ({(1 - np.mean(y)) * 100:.2f}%)")

# 计算正类比例
pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0
print(f"  正类权重比例: {pos_weight:.2f}")

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

print(f"总共 {len(models)} 个模型将被测试")
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

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_combined_{timestamp}"
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

    fold = 0
    successful_folds = 0

    for train_idx, test_idx in skf.split(X, y):
        fold += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                y_proba = expit(decision_scores)
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_proba = y_pred.astype(float)

            # 获取最优阈值（对于双优化器）
            if isinstance(model, DualOptimizerClassifier):
                optimal_threshold = model.optimal_threshold
            else:
                optimal_threshold = 0.5

            # 评估指标
            metrics = evaluate_comprehensive_metrics(y_test, y_pred, y_proba, model_name)

            # 存储指标
            for key in fold_metrics.keys():
                if key in metrics:
                    fold_metrics[key].append(metrics[key])

            fold_metrics['optimal_threshold'].append(optimal_threshold)

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
                           'Linear') and not m.startswith('Quadratic')]

    if dual_models and baseline_models:
        dual_pr_auc = np.mean([results[m].get('mean_pr_auc', 0) for m in dual_models])
        baseline_pr_auc = np.mean([results[m].get('mean_pr_auc', 0) for m in baseline_models])

        print(f"   - 双提升模型平均PR AUC: {dual_pr_auc:.4f}")
        print(f"   - 传统模型平均PR AUC: {baseline_pr_auc:.4f}")
        improvement = ((dual_pr_auc - baseline_pr_auc) / baseline_pr_auc * 100) if baseline_pr_auc > 0 else 0
        print(f"   - 平均提升: {improvement:.1f}%")

    print(f"\n4. 临床推荐:")
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