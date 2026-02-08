
# PR曲线最大值的版本

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    precision_score, roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve, accuracy_score,
    recall_score, confusion_matrix, f1_score, fbeta_score, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
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


# ========== 自定义分类器：PR AUC优化 ==========
class PRAUCOptimizedClassifier(BaseEstimator, ClassifierMixin):
    """专门优化PR AUC的分类器"""

    def __init__(self, base_model, beta=0.8, n_thresholds=81):
        self.base_model = base_model
        self.beta = beta  # F-beta中的beta参数
        self.n_thresholds = n_thresholds
        self.optimal_threshold = 0.5
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        """训练并优化阈值"""
        if X_val is None or y_val is None:
            # 如果没有提供验证集，从训练集中划分
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            X_train, y_train = X, y

        # 训练基础模型
        self.base_model.fit(X_train, y_train)

        # 获取验证集概率
        if hasattr(self.base_model, 'predict_proba'):
            y_val_proba = self.base_model.predict_proba(X_val)[:, 1]
        elif hasattr(self.base_model, 'decision_function'):
            decision_scores = self.base_model.decision_function(X_val)
            y_val_proba = expit(decision_scores)
        else:
            y_val_proba = None

        # 优化阈值以最大化平衡分数
        if y_val_proba is not None:
            thresholds = np.linspace(0.1, 0.9, self.n_thresholds)
            best_score = -1

            for threshold in thresholds:
                y_val_pred = (y_val_proba >= threshold).astype(int)

                # 计算F-beta分数
                f_beta = fbeta_score(y_val, y_val_pred, beta=self.beta, zero_division=0)

                # 同时考虑精确率和召回率
                precision = precision_score(y_val, y_val_pred, zero_division=0)
                recall = recall_score(y_val, y_val_pred, zero_division=0)

                # 计算几何平均
                geometric_mean = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0

                # 综合平衡分数
                balanced_score = 0.4 * f_beta + 0.3 * precision + 0.2 * recall + 0.1 * geometric_mean

                if balanced_score > best_score:
                    best_score = balanced_score
                    self.optimal_threshold = threshold

        # 在整个训练集上重新训练
        self.base_model.fit(X, y)
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
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        elif hasattr(self.base_model, 'decision_function'):
            decision_scores = self.base_model.decision_function(X)
            proba_positive = expit(decision_scores)
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            raise AttributeError("基础模型没有predict_proba或decision_function方法")

    def get_optimal_threshold(self):
        return self.optimal_threshold

    def get_params(self, deep=True):
        return {'base_model': self.base_model, 'beta': self.beta, 'n_thresholds': self.n_thresholds}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ========== 自定义分类器：代价敏感随机森林 ==========
class CostSensitiveRandomForest(RandomForestClassifier):
    """代价敏感的随机森林"""

    def __init__(self, cost_ratio=1.5, **kwargs):
        super().__init__(**kwargs)
        self.cost_ratio = cost_ratio

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        # 调整阈值：cost_ratio > 1 表示假阴性代价更高
        threshold = 0.5 / self.cost_ratio if self.cost_ratio > 0 else 0.5
        return (proba >= threshold).astype(int)


# ========== 自定义分类器：平衡决策树 ==========
class BalancedDecisionTree(DecisionTreeClassifier):
    """平衡精确率和召回率的决策树"""

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # 平衡参数：0-1之间，越大越重视召回率

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        # 根据alpha调整阈值
        threshold = 0.5 * (1 - self.alpha) + 0.3 * self.alpha
        return (proba >= threshold).astype(int)


# ========== 数据加载 ==========
print("=" * 80)
print("加载数据...")
print("=" * 80)

data = pd.read_csv('SAP65.csv')
np.set_printoptions(suppress=True)
y = data["SAP"].values
X = data.iloc[:, 0:39].values

print(f"数据集信息:")
print(f"  总样本数: {len(y)}")
print(f"  特征维度: {X.shape[1]}")
print(f"  正类(SAP)数量: {sum(y)} ({np.mean(y) * 100:.2f}%)")
print(f"  负类数量: {len(y) - sum(y)} ({(1 - np.mean(y)) * 100:.2f}%)")

# 计算正类比例，用于处理不平衡数据
pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0
print(f"  正类权重比例: {pos_weight:.2f}")

# ========== 特征工程：添加有助于平衡PR的特征 ==========
print(f"\n{'=' * 80}")
print("特征工程...")
print("=" * 80)


def create_balanced_features(X_original):
    """创建可能有助于平衡精确率和召回率的特征"""
    X = np.copy(X_original)
    n_features = X.shape[1]
    print(f"原始特征维度: {n_features}")

    new_features = []
    feature_names = []

    # 添加重要的交互特征（假设前5个特征最重要）
    if n_features >= 5:
        for i in range(min(5, n_features)):
            for j in range(i + 1, min(5, n_features)):
                interaction = X[:, i] * X[:, j]
                new_features.append(interaction.reshape(-1, 1))
                feature_names.append(f"inter_{i}_{j}")

        # 添加平方项（非线性）
        for i in range(min(5, n_features)):
            squared = X[:, i] ** 2
            new_features.append(squared.reshape(-1, 1))
            feature_names.append(f"sq_{i}")

        # 添加比率特征
        if n_features >= 4:
            for i in range(min(4, n_features)):
                for j in range(i + 1, min(4, n_features)):
                    if np.min(np.abs(X[:, j])) > 0.001:  # 避免除以零
                        ratio = X[:, i] / (X[:, j] + 0.001)
                        new_features.append(ratio.reshape(-1, 1))
                        feature_names.append(f"ratio_{i}_{j}")

    if new_features:
        X_enhanced = np.hstack([X] + new_features)
        print(f"增强后特征维度: {X_enhanced.shape[1]}")
        print(f"添加了 {len(new_features)} 个新特征")
        return X_enhanced

    return X


# 应用特征工程
X = create_balanced_features(X)

# ========== 定义所有要比较的模型（PR AUC优化版本） ==========
print(f"\n{'=' * 80}")
print("初始化模型...")
print("=" * 80)

models = {}

# 1. 基础线性模型
models["Logistic_Basic"] = LogisticRegression(
    max_iter=5000,
    class_weight='balanced',
    random_state=42
)

# 2. PR AUC优化的逻辑回归
models["Logistic_PRAUC"] = PRAUCOptimizedClassifier(
    base_model=LogisticRegression(
        max_iter=5000,
        class_weight='balanced',
        C=0.5,
        solver='liblinear',
        random_state=42
    ),
    beta=0.8
)

# 3. 支持向量机
models["SVM_RBF"] = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

# 4. 决策树
models["DecisionTree"] = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)

# 5. 平衡决策树
models["Balanced_DT"] = BalancedDecisionTree(
    alpha=0.4,
    class_weight='balanced',
    random_state=42,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10
)

# 6. 随机森林
models["RandomForest"] = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 7. 代价敏感随机森林
models["CostSensitive_RF"] = CostSensitiveRandomForest(
    cost_ratio=1.5,
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    max_depth=15,
    min_samples_leaf=5
)

# 8. PR AUC优化的随机森林
models["RandomForest_PRAUC"] = PRAUCOptimizedClassifier(
    base_model=RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        max_depth=15,
        min_samples_leaf=3,
        n_jobs=-1
    ),
    beta=0.8
)

# 9. Extra Trees
models["ExtraTrees"] = ExtraTreesClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 10. 梯度提升
models["GradientBoosting"] = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# 11. PR AUC优化的梯度提升
models["GradientBoosting_PRAUC"] = PRAUCOptimizedClassifier(
    base_model=GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    beta=0.85
)

# 12. AdaBoost
models["AdaBoost"] = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

# 13. Bagging
models["Bagging_DT"] = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(class_weight='balanced'),
    n_estimators=50,
    random_state=42
)

# 14. 神经网络
models["MLP"] = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    random_state=42
)

# 15. 贝叶斯
models["GaussianNB"] = GaussianNB()

# 16. 判别分析
models["LinearDiscriminant"] = LinearDiscriminantAnalysis()
models["QuadraticDiscriminant"] = QuadraticDiscriminantAnalysis()

# 17. K近邻
models["KNN_5"] = KNeighborsClassifier(n_neighbors=5)
models["KNN_10"] = KNeighborsClassifier(n_neighbors=10)

# 18. XGBoost（如果可用）
if XGB_AVAILABLE:
    models["XGBoost_Basic"] = XGBClassifier(
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
    )

    models["XGBoost_PRAUC"] = PRAUCOptimizedClassifier(
        base_model=XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight * 1.2,
            reg_lambda=1,
            reg_alpha=0.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        beta=0.7
    )

# 19. 堆叠集成
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

if XGB_AVAILABLE:
    base_estimators.insert(1, (
        'xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')))

models["Stacking"] = StackingClassifier(
    estimators=base_estimators[:3],
    final_estimator=LogisticRegression(),
    cv=3
)

# 20. 投票集成
if XGB_AVAILABLE:
    voting_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=pos_weight, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]
else:
    voting_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42))
    ]

models["Voting"] = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
    weights=[1.2, 1.5, 1.0] if XGB_AVAILABLE else [1.2, 1.0, 0.8],
    n_jobs=-1
)

# 21. 校准模型
print("添加校准模型...")
for calib_method in ['sigmoid', 'isotonic']:
    models[f"LR_Calibrated_{calib_method}"] = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        method=calib_method,
        cv=3
    )

    models[f"RF_Calibrated_{calib_method}"] = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42),
        method=calib_method,
        cv=3
    )

print(f"总共 {len(models)} 个模型将被测试")
print("模型列表:", list(models.keys()))


# ========== PR AUC优化阈值函数 ==========
def optimize_threshold_for_prauc(y_true, y_pred_proba, beta=0.8):
    """优化阈值以最大化平衡指标"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_score = -1
    best_threshold = 0.5
    best_metrics = {}

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if precision + recall > 0:
            # 计算F-beta分数
            f_beta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

            # 计算几何平均
            geometric_mean = np.sqrt(precision * recall)

            # 综合平衡分数
            balanced_score = 0.4 * f_beta + 0.3 * precision + 0.2 * recall + 0.1 * geometric_mean

            if balanced_score > best_score:
                best_score = balanced_score
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f_beta': f_beta,
                    'geometric_mean': geometric_mean,
                    'balanced_score': balanced_score
                }

    return best_threshold, best_metrics


# ========== 设置交叉验证 ==========
n_folds = 30
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ========== 存储所有模型的结果 ==========
results = {}
all_fold_results = []  # 存储每一折的详细结果
model_summaries = []  # 存储模型汇总结果

# ========== 为ROC和PR曲线存储数据 ==========
all_roc_data = {}
all_pr_data = {}

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_prauc_optimized_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ========== 对每个模型进行训练和评估 ==========
print(f"\n{'=' * 80}")
print("开始模型训练和评估...")
print("=" * 80)

for model_idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"\n{'=' * 80}")
    print(f"正在训练模型 [{model_idx}/{len(models)}]: {model_name}")
    print('=' * 80)

    # 存储当前模型的指标
    model_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'f_beta': [],  # F-beta分数
        'roc_auc': [],
        'pr_auc': [],
        'tpr': [],  # sensitivity
        'tnr': [],  # specificity
        'ppv': [],  # precision
        'npv': [],  # negative predictive value
        'optimal_threshold': [],  # 最优阈值
        'balanced_score': [],  # 平衡分数
        'predictions': [],
        'probabilities': [],
        'fold_indices': [],
        'y_true': []
    }

    # 存储ROC曲线数据
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 存储PR曲线数据
    precisions = []
    pr_aucs = []
    mean_recall = np.linspace(0, 1, 100)

    fold = 0
    successful_folds = 0

    for train_idx, test_idx in skf.split(X, y):
        fold += 1

        # 划分训练集和测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 进一步划分训练集为训练和验证（用于阈值优化）
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=42
        )

        try:
            # 训练模型
            if isinstance(model, PRAUCOptimizedClassifier):
                # PR AUC优化模型有自己的fit方法
                model.fit(X_train_main, y_train_main, X_val, y_val)
                optimal_threshold = model.get_optimal_threshold()
            else:
                # 其他模型
                model.fit(X_train, y_train)

                # 获取验证集概率以优化阈值
                if hasattr(model, 'predict_proba'):
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_val)
                    y_val_proba = expit(decision_scores)
                else:
                    y_val_proba = None

                # 优化阈值
                if y_val_proba is not None:
                    optimal_threshold, _ = optimize_threshold_for_prauc(y_val, y_val_proba)
                else:
                    optimal_threshold = 0.5

            # 获取测试集概率
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                y_pred_proba = expit(decision_scores)
            else:
                y_pred_proba = None

            # 使用最优阈值进行预测
            if y_pred_proba is not None:
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred.astype(float)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            f_beta = fbeta_score(y_test, y_pred, beta=0.8, zero_division=0)

            # 计算平衡分数
            geometric_mean = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
            balanced_score = 0.4 * f_beta + 0.3 * precision + 0.2 * recall + 0.1 * geometric_mean

            # ROC曲线和AUC
            if len(np.unique(y_pred_proba)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                model_results['roc_auc'].append(roc_auc)

                # 存储ROC曲线数据
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)
            else:
                roc_auc = 0.5
                model_results['roc_auc'].append(roc_auc)
                tprs.append(np.zeros_like(mean_fpr))
                aucs.append(0.5)

            # PR曲线和AUC
            if len(np.unique(y_pred_proba)) > 1:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall_vals, precision_vals)
                model_results['pr_auc'].append(pr_auc)

                # 存储PR曲线数据
                precision_interp = np.interp(mean_recall, recall_vals[::-1], precision_vals[::-1])
                precisions.append(precision_interp)
                pr_aucs.append(pr_auc)
            else:
                pr_auc = np.mean(y_test)
                model_results['pr_auc'].append(pr_auc)
                precisions.append(np.ones_like(mean_recall) * np.mean(y_test))
                pr_aucs.append(np.mean(y_test))

            # 混淆矩阵相关指标
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                sensitivity = specificity = ppv = npv = 0

            # 存储每一折的结果
            model_results['accuracy'].append(accuracy)
            model_results['precision'].append(precision)
            model_results['recall'].append(recall)
            model_results['f1'].append(f1)
            model_results['f_beta'].append(f_beta)
            model_results['tpr'].append(sensitivity)
            model_results['tnr'].append(specificity)
            model_results['ppv'].append(ppv)
            model_results['npv'].append(npv)
            model_results['optimal_threshold'].append(optimal_threshold)
            model_results['balanced_score'].append(balanced_score)
            model_results['predictions'].append(y_pred)
            model_results['probabilities'].append(y_pred_proba)
            model_results['fold_indices'].append((train_idx, test_idx))
            model_results['y_true'].append(y_test)

            # 存储到全局的每一折结果列表
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'F_Beta_Score': f_beta,
                'Balanced_Score': balanced_score,
                'Geometric_Mean': geometric_mean,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc,
                'Sensitivity_TPR': sensitivity,
                'Specificity_TNR': specificity,
                'PPV': ppv,
                'NPV': npv,
                'Optimal_Threshold': optimal_threshold,
                'Train_Samples': len(train_idx),
                'Test_Samples': len(test_idx),
                'Test_Positive_Rate': np.mean(y_test),
                'True_Positives': tp if cm.shape == (2, 2) else 0,
                'False_Negatives': fn if cm.shape == (2, 2) else 0,
                'False_Positives': fp if cm.shape == (2, 2) else 0,
                'True_Negatives': tn if cm.shape == (2, 2) else 0,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_fold_results.append(fold_result)

            successful_folds += 1

            # 每5个fold打印一次进度
            if fold % 5 == 0:
                print(f"  已完成 {fold}/{n_folds} folds, 平衡分数: {balanced_score:.3f}, PR AUC: {pr_auc:.3f}")

        except Exception as e:
            print(f"  Fold {fold} 失败: {str(e)[:100]}...")

            # 即使失败也记录
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1_Score': np.nan,
                'F_Beta_Score': np.nan,
                'Balanced_Score': np.nan,
                'Geometric_Mean': np.nan,
                'ROC_AUC': np.nan,
                'PR_AUC': np.nan,
                'Sensitivity_TPR': np.nan,
                'Specificity_TNR': np.nan,
                'PPV': np.nan,
                'NPV': np.nan,
                'Optimal_Threshold': np.nan,
                'Train_Samples': len(train_idx),
                'Test_Samples': len(test_idx),
                'Test_Positive_Rate': np.mean(y_test),
                'True_Positives': np.nan,
                'False_Negatives': np.nan,
                'False_Positives': np.nan,
                'True_Negatives': np.nan,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Error': str(e)[:200]
            }
            all_fold_results.append(fold_result)
            continue

    # 检查是否有成功的fold
    if successful_folds == 0:
        print(f"模型 {model_name} 所有fold都失败，跳过")

        # 记录汇总结果（即使全部失败）
        summary_result = {
            'Model': model_name,
            'Total_Folds': n_folds,
            'Successful_Folds': 0,
            'Success_Rate': 0,
            'Mean_Accuracy': np.nan,
            'Std_Accuracy': np.nan,
            'Mean_Precision': np.nan,
            'Std_Precision': np.nan,
            'Mean_Recall': np.nan,
            'Std_Recall': np.nan,
            'Mean_F1': np.nan,
            'Std_F1': np.nan,
            'Mean_F_Beta': np.nan,
            'Std_F_Beta': np.nan,
            'Mean_Balanced_Score': np.nan,
            'Std_Balanced_Score': np.nan,
            'Mean_ROC_AUC': np.nan,
            'Std_ROC_AUC': np.nan,
            'Mean_PR_AUC': np.nan,
            'Std_PR_AUC': np.nan,
            'Mean_Sensitivity': np.nan,
            'Std_Sensitivity': np.nan,
            'Mean_Specificity': np.nan,
            'Std_Specificity': np.nan,
            'Mean_PPV': np.nan,
            'Std_PPV': np.nan,
            'Mean_NPV': np.nan,
            'Std_NPV': np.nan,
            'Mean_Optimal_Threshold': np.nan,
            'Std_Optimal_Threshold': np.nan,
            'Min_PR_AUC': np.nan,
            'Max_PR_AUC': np.nan,
            'Median_PR_AUC': np.nan,
            'Status': 'FAILED'
        }
        model_summaries.append(summary_result)
        continue

    # 计算平均指标和标准差
    metrics_to_calculate = [
        'accuracy', 'precision', 'recall', 'f1', 'f_beta', 'balanced_score',
        'roc_auc', 'pr_auc', 'tpr', 'tnr', 'ppv', 'npv', 'optimal_threshold'
    ]

    for metric in metrics_to_calculate:
        if metric in model_results and len(model_results[metric]) > 0:
            model_results[f'mean_{metric}'] = np.nanmean(model_results[metric])
            model_results[f'std_{metric}'] = np.nanstd(model_results[metric])

    # 计算额外的PR AUC统计量
    if len(model_results['pr_auc']) > 0:
        model_results['min_pr_auc'] = np.nanmin(model_results['pr_auc'])
        model_results['max_pr_auc'] = np.nanmax(model_results['pr_auc'])
        model_results['median_pr_auc'] = np.nanmedian(model_results['pr_auc'])

    # 存储ROC曲线数据
    if len(tprs) > 0:
        all_roc_data[model_name] = {
            'mean_fpr': mean_fpr,
            'mean_tpr': np.nanmean(tprs, axis=0),
            'std_tpr': np.nanstd(tprs, axis=0),
            'mean_auc': np.nanmean(aucs),
            'std_auc': np.nanstd(aucs),
            'tprs': tprs
        }

    # 存储PR曲线数据
    if len(precisions) > 0:
        all_pr_data[model_name] = {
            'mean_recall': mean_recall,
            'mean_precision': np.nanmean(precisions, axis=0),
            'std_precision': np.nanstd(precisions, axis=0),
            'mean_pr_auc': np.nanmean(pr_aucs),
            'std_pr_auc': np.nanstd(pr_aucs),
            'precisions': precisions
        }

    # 存储当前模型的结果
    results[model_name] = model_results

    # 创建模型汇总结果
    summary_result = {
        'Model': model_name,
        'Total_Folds': n_folds,
        'Successful_Folds': successful_folds,
        'Success_Rate': successful_folds / n_folds * 100,
        'Mean_Accuracy': model_results.get('mean_accuracy', np.nan),
        'Std_Accuracy': model_results.get('std_accuracy', np.nan),
        'Mean_Precision': model_results.get('mean_precision', np.nan),
        'Std_Precision': model_results.get('std_precision', np.nan),
        'Mean_Recall': model_results.get('mean_recall', np.nan),
        'Std_Recall': model_results.get('std_recall', np.nan),
        'Mean_F1': model_results.get('mean_f1', np.nan),
        'Std_F1': model_results.get('std_f1', np.nan),
        'Mean_F_Beta': model_results.get('mean_f_beta', np.nan),
        'Std_F_Beta': model_results.get('std_f_beta', np.nan),
        'Mean_Balanced_Score': model_results.get('mean_balanced_score', np.nan),
        'Std_Balanced_Score': model_results.get('std_balanced_score', np.nan),
        'Mean_ROC_AUC': model_results.get('mean_roc_auc', np.nan),
        'Std_ROC_AUC': model_results.get('std_roc_auc', np.nan),
        'Mean_PR_AUC': model_results.get('mean_pr_auc', np.nan),
        'Std_PR_AUC': model_results.get('std_pr_auc', np.nan),
        'Mean_Sensitivity': model_results.get('mean_tpr', np.nan),
        'Std_Sensitivity': model_results.get('std_tpr', np.nan),
        'Mean_Specificity': model_results.get('mean_tnr', np.nan),
        'Std_Specificity': model_results.get('std_tnr', np.nan),
        'Mean_PPV': model_results.get('mean_ppv', np.nan),
        'Std_PPV': model_results.get('std_ppv', np.nan),
        'Mean_NPV': model_results.get('mean_npv', np.nan),
        'Std_NPV': model_results.get('std_npv', np.nan),
        'Mean_Optimal_Threshold': model_results.get('mean_optimal_threshold', np.nan),
        'Std_Optimal_Threshold': model_results.get('std_optimal_threshold', np.nan),
        'Min_PR_AUC': model_results.get('min_pr_auc', np.nan),
        'Max_PR_AUC': model_results.get('max_pr_auc', np.nan),
        'Median_PR_AUC': model_results.get('median_pr_auc', np.nan),
        'Status': 'SUCCESS'
    }
    model_summaries.append(summary_result)

    # 打印当前模型的性能
    print(f"\n{model_name} 性能汇总 ({successful_folds}/{n_folds} folds成功):")
    print(
        f"  PR AUC:          {model_results.get('mean_pr_auc', np.nan):.4f} ± {model_results.get('std_pr_auc', np.nan):.4f}")
    print(
        f"  平衡分数:        {model_results.get('mean_balanced_score', np.nan):.4f} ± {model_results.get('std_balanced_score', np.nan):.4f}")
    print(
        f"  F-beta分数:      {model_results.get('mean_f_beta', np.nan):.4f} ± {model_results.get('std_f_beta', np.nan):.4f}")
    print(
        f"  召回率(Recall):  {model_results.get('mean_recall', np.nan):.4f} ± {model_results.get('std_recall', np.nan):.4f}")
    print(
        f"  精确率(Precision): {model_results.get('mean_precision', np.nan):.4f} ± {model_results.get('std_precision', np.nan):.4f}")
    print(f"  F1分数:         {model_results.get('mean_f1', np.nan):.4f} ± {model_results.get('std_f1', np.nan):.4f}")
    print(
        f"  最优阈值:       {model_results.get('mean_optimal_threshold', np.nan):.4f} ± {model_results.get('std_optimal_threshold', np.nan):.4f}")

# ========== 保存结果到CSV文件 ==========
print(f"\n{'=' * 80}")
print("正在保存结果到CSV文件...")
print('=' * 80)

# 1. 保存每一折的详细结果
if all_fold_results:
    fold_df = pd.DataFrame(all_fold_results)
    fold_csv_path = os.path.join(results_dir, f'fold_results_detailed.csv')
    fold_df.to_csv(fold_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存每一折详细结果到: {fold_csv_path}")
    print(f"  共 {len(fold_df)} 条记录，{len(fold_df.columns)} 个指标")

# 2. 保存模型汇总结果
if model_summaries:
    summary_df = pd.DataFrame(model_summaries)
    summary_csv_path = os.path.join(results_dir, f'model_summary_results.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存模型汇总结果到: {summary_csv_path}")
    print(f"  共 {len(summary_df)} 个模型，{len(summary_df.columns)} 个指标")

# 3. 创建详细排名表
if len(results) > 0:
    # 准备详细汇总数据
    detailed_summary = []
    for model_name, model_results in results.items():
        if 'mean_pr_auc' not in model_results:
            continue

        # 从汇总表中获取成功fold数
        success_info = next((item for item in model_summaries if item['Model'] == model_name), {})

        # 计算平衡比
        mean_precision = model_results.get('mean_precision', 0)
        mean_recall = model_results.get('mean_recall', 0)
        balance_ratio = mean_precision / mean_recall if mean_recall > 0 else np.inf

        model_summary = {
            'Model': model_name,
            'Total_Folds_Attempted': n_folds,
            'Successful_Folds': success_info.get('Successful_Folds', 0),
            'Success_Rate': success_info.get('Success_Rate', 0),
            'Mean_PR_AUC': model_results.get('mean_pr_auc', np.nan),
            'Std_PR_AUC': model_results.get('std_pr_auc', np.nan),
            'Mean_Balanced_Score': model_results.get('mean_balanced_score', np.nan),
            'Std_Balanced_Score': model_results.get('std_balanced_score', np.nan),
            'Mean_ROC_AUC': model_results.get('mean_roc_auc', np.nan),
            'Std_ROC_AUC': model_results.get('std_roc_auc', np.nan),
            'Mean_Precision': mean_precision,
            'Std_Precision': model_results.get('std_precision', np.nan),
            'Mean_Recall': mean_recall,
            'Std_Recall': model_results.get('std_recall', np.nan),
            'Balance_Ratio': balance_ratio,
            'Mean_F1_Score': model_results.get('mean_f1', np.nan),
            'Std_F1_Score': model_results.get('std_f1', np.nan),
            'Mean_F_Beta_Score': model_results.get('mean_f_beta', np.nan),
            'Std_F_Beta_Score': model_results.get('std_f_beta', np.nan),
            'Mean_Optimal_Threshold': model_results.get('mean_optimal_threshold', np.nan),
            'Std_Optimal_Threshold': model_results.get('std_optimal_threshold', np.nan),
            'Min_PR_AUC': model_results.get('min_pr_auc', np.nan),
            'Max_PR_AUC': model_results.get('max_pr_auc', np.nan),
            'Median_PR_AUC': model_results.get('median_pr_auc', np.nan),
            'Data_Positive_Rate': np.mean(y) * 100,
            'Evaluation_Date': datetime.now().strftime("%Y-%m-%d")
        }
        detailed_summary.append(model_summary)

    if detailed_summary:
        detailed_summary_df = pd.DataFrame(detailed_summary)

        # 按PR AUC排序
        prauc_ranked_df = detailed_summary_df.sort_values('Mean_PR_AUC', ascending=False)
        prauc_rank_path = os.path.join(results_dir, f'model_ranking_by_prauc.csv')
        prauc_ranked_df.to_csv(prauc_rank_path, index=False, encoding='utf-8-sig')
        print(f"已保存按PR AUC排名到: {prauc_rank_path}")

        # 按平衡分数排序
        balanced_ranked_df = detailed_summary_df.sort_values('Mean_Balanced_Score', ascending=False)
        balanced_rank_path = os.path.join(results_dir, f'model_ranking_by_balanced_score.csv')
        balanced_ranked_df.to_csv(balanced_rank_path, index=False, encoding='utf-8-sig')
        print(f"已保存按平衡分数排名到: {balanced_rank_path}")

        # 按F-beta分数排序
        fbeta_ranked_df = detailed_summary_df.sort_values('Mean_F_Beta_Score', ascending=False)
        fbeta_rank_path = os.path.join(results_dir, f'model_ranking_by_fbeta.csv')
        fbeta_ranked_df.to_csv(fbeta_rank_path, index=False, encoding='utf-8-sig')
        print(f"已保存按F-beta分数排名到: {fbeta_rank_path}")

        # 显示前10个模型（按PR AUC）
        print(f"\n{'=' * 80}")
        print("模型PR AUC排名 (前10):")
        print('=' * 80)

        display_cols = ['Model', 'Mean_PR_AUC', 'Mean_Balanced_Score', 'Mean_F_Beta_Score',
                        'Mean_Precision', 'Mean_Recall', 'Mean_F1_Score', 'Mean_Optimal_Threshold']
        print(prauc_ranked_df[display_cols].head(10).to_string(index=False))

# ========== 绘制ROC曲线 ==========
if len(results) > 0 and len(all_roc_data) > 0:
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

            std_tpr = roc_info['std_tpr']
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                             color=colors[idx], alpha=0.1)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves Comparison - PR AUC Optimized Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, 'ROC_Curves_PRAUC_Optimized.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n已保存ROC曲线图到: {roc_plot_path}")

# ========== 绘制PR曲线 ==========
if len(results) > 0 and len(all_pr_data) > 0:
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

            std_precision = pr_info['std_precision']
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            plt.fill_between(mean_recall, precision_lower, precision_upper,
                             color=colors[idx], alpha=0.1)

    plt.axhline(y=pos_rate, color='gray', linestyle='--', lw=1, alpha=0.5,
                label=f'Random (AP = {pos_rate:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curves - Top PR AUC Models', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(results_dir, 'PR_Curves_Top_PRAUC.png')
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存PR曲线图到: {pr_plot_path}")

# ========== 绘制PR平衡分析图 ==========
if len(results) > 0:
    # 准备数据
    model_names = []
    prauc_scores = []
    balanced_scores = []
    precisions = []
    recalls = []
    thresholds = []

    for model_name, model_results in results.items():
        if 'mean_pr_auc' in model_results:
            model_names.append(model_name)
            prauc_scores.append(model_results['mean_pr_auc'])
            balanced_scores.append(model_results.get('mean_balanced_score', 0))
            precisions.append(model_results.get('mean_precision', 0))
            recalls.append(model_results.get('mean_recall', 0))
            thresholds.append(model_results.get('mean_optimal_threshold', 0.5))

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PR AUC分布图
    ax1 = axes[0, 0]
    sorted_indices = np.argsort(prauc_scores)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices[:10]]
    sorted_prauc = [prauc_scores[i] for i in sorted_indices[:10]]

    bars = ax1.barh(range(len(sorted_names)), sorted_prauc, color='lightgreen')
    ax1.set_yticks(range(len(sorted_names)))
    ax1.set_yticklabels(sorted_names, fontsize=9)
    ax1.set_xlabel('PR AUC', fontsize=11)
    ax1.set_title('Top 10 Models by PR AUC', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 在条上添加数值
    for i, (bar, prauc_val) in enumerate(zip(bars, sorted_prauc)):
        ax1.text(prauc_val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{prauc_val:.3f}', va='center', fontsize=8)

    # 2. 精确率vs召回率散点图
    ax2 = axes[0, 1]
    scatter = ax2.scatter(recalls, precisions, s=50, c=prauc_scores,
                          cmap='RdYlGn', alpha=0.7, edgecolors='k')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision vs Recall (Color = PR AUC)', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    # 添加对角线
    ax2.plot([0, 1], [1, 0], 'k--', alpha=0.3)

    # 添加颜色条
    plt.colorbar(scatter, ax=ax2, label='PR AUC')

    # 3. 阈值vs平衡分数散点图
    ax3 = axes[1, 0]
    bubble_sizes = [min(max(bs * 300, 50), 500) for bs in balanced_scores]

    scatter2 = ax3.scatter(thresholds, prauc_scores, s=bubble_sizes,
                           c=balanced_scores, cmap='coolwarm', alpha=0.6, edgecolors='k')
    ax3.set_xlabel('Optimal Threshold', fontsize=11)
    ax3.set_ylabel('PR AUC', fontsize=11)
    ax3.set_title('Threshold vs PR AUC (Bubble Size = Balanced Score)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 添加颜色条
    plt.colorbar(scatter2, ax=ax3, label='Balanced Score')

    # 4. 模型性能热图
    ax4 = axes[1, 1]
    # 选择前8个模型
    top_n = min(8, len(model_names))
    top_indices = sorted_indices[:top_n]

    performance_data = []
    for idx in top_indices:
        perf_row = [
            prauc_scores[idx],
            balanced_scores[idx],
            precisions[idx],
            recalls[idx],
            thresholds[idx]
        ]
        performance_data.append(perf_row)

    performance_matrix = np.array(performance_data).T
    im = ax4.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')

    # 设置坐标轴
    ax4.set_xticks(np.arange(top_n))
    ax4.set_xticklabels([model_names[i] for i in top_indices], rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(np.arange(5))
    ax4.set_yticklabels(['PR AUC', 'Balanced Score', 'Precision', 'Recall', 'Threshold'], fontsize=10)

    # 添加数值
    for i in range(5):
        for j in range(top_n):
            value = performance_matrix[i, j]
            text = ax4.text(j, i, f'{value:.3f}', ha="center", va="center", color="black", fontsize=8)

    ax4.set_title('Top Models Performance Heatmap', fontsize=12, fontweight='bold')

    plt.tight_layout()
    analysis_plot_path = os.path.join(results_dir, 'PRAUC_Analysis_Plots.png')
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存PR AUC分析图到: {analysis_plot_path}")

# ========== 生成最终PR AUC优化报告 ==========
print(f"\n{'=' * 80}")
print("PR AUC优化最终报告")
print('=' * 80)

if len(results) > 0:
    # 计算总体统计
    all_prauc = [r.get('mean_pr_auc', 0) for r in results.values()]
    all_balanced = [r.get('mean_balanced_score', 0) for r in results.values()]
    all_precision = [r.get('mean_precision', 0) for r in results.values()]
    all_recall = [r.get('mean_recall', 0) for r in results.values()]

    avg_prauc = np.nanmean(all_prauc)
    max_prauc = np.nanmax(all_prauc)
    min_prauc = np.nanmin(all_prauc)

    print(f"PR AUC统计:")
    print(f"  平均值: {avg_prauc:.4f}")
    print(f"  最大值: {max_prauc:.4f}")
    print(f"  最小值: {min_prauc:.4f}")
    print(f"  范围:   {max_prauc - min_prauc:.4f}")

    # 找出最佳模型
    best_by_prauc = max(results.items(), key=lambda x: x[1].get('mean_pr_auc', 0))
    best_by_balanced = max(results.items(), key=lambda x: x[1].get('mean_balanced_score', 0))

    best_prauc_name, best_prauc_results = best_by_prauc
    best_balanced_name, best_balanced_results = best_by_balanced

    print(f"\n最佳PR AUC模型: {best_prauc_name}")
    print(f"  PR AUC:         {best_prauc_results.get('mean_pr_auc', np.nan):.4f}")
    print(f"  平衡分数:       {best_prauc_results.get('mean_balanced_score', np.nan):.4f}")
    print(f"  精确率:         {best_prauc_results.get('mean_precision', np.nan):.4f}")
    print(f"  召回率:         {best_prauc_results.get('mean_recall', np.nan):.4f}")
    print(f"  F1分数:         {best_prauc_results.get('mean_f1', np.nan):.4f}")
    print(f"  最优阈值:       {best_prauc_results.get('mean_optimal_threshold', np.nan):.4f}")

    print(f"\n最佳平衡模型: {best_balanced_name}")
    print(f"  平衡分数:       {best_balanced_results.get('mean_balanced_score', np.nan):.4f}")
    print(f"  PR AUC:         {best_balanced_results.get('mean_pr_auc', np.nan):.4f}")
    print(f"  精确率:         {best_balanced_results.get('mean_precision', np.nan):.4f}")
    print(f"  召回率:         {best_balanced_results.get('mean_recall', np.nan):.4f}")

    # 临床建议
    precision_val = best_prauc_results.get('mean_precision', 0)
    recall_val = best_prauc_results.get('mean_recall', 0)
    prauc_val = best_prauc_results.get('mean_pr_auc', 0)

    print(f"\n临床建议:")

    # PR AUC评估
    if prauc_val > 0.7:
        print("  ✓ 模型具有优秀的PR AUC - 整体性能良好")
    elif prauc_val > 0.5:
        print("  ⚠ 模型PR AUC中等 - 可考虑进一步优化")
    else:
        print("  ✗ 模型PR AUC较低 - 需要重新设计")

    # 平衡性评估
    balance_ratio = precision_val / recall_val if recall_val > 0 else np.inf
    if 0.8 < balance_ratio < 1.25:
        print("  ✓ 精确率和召回率平衡良好")
    elif balance_ratio > 1.25:
        print("  ⚠ 模型偏向高精确率（假阳性少，但可能漏诊）")
    else:
        print("  ⚠ 模型偏向高召回率（漏诊少，但假阳性多）")

    # 阈值建议
    optimal_threshold = best_prauc_results.get('mean_optimal_threshold', 0.5)
    print(f"\n阈值调整建议:")
    print(f"  当前最优阈值: {optimal_threshold:.3f}")
    if optimal_threshold < 0.4:
        print("  → 当前阈值较低，模型偏向高召回率")
        print("  → 如需更高精确率，可尝试提高阈值至0.4-0.5")
    elif optimal_threshold > 0.6:
        print("  → 当前阈值较高，模型偏向高精确率")
        print("  → 如需更高召回率，可尝试降低阈值至0.4-0.5")
    else:
        print("  → 当前阈值适中，平衡了精确率和召回率")

print(f"\n{'=' * 80}")
print("所有结果已保存完毕！")
print(f"结果保存在目录: {results_dir}")
print('=' * 80)

# 显示保存的文件列表
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f"\n目录中包含以下文件:")
    for file in sorted(files):
        file_path = os.path.join(results_dir, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024  # 转换为KB
            print(f"  - {file} ({file_size:.1f} KB)")