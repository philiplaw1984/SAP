
# 增加多种模型，保存每一折结果

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, roc_auc_score, average_precision_score,
    roc_curve, auc, precision_recall_curve, accuracy_score,
    recall_score, confusion_matrix, f1_score
)
import os
from datetime import datetime

# 只导入XGBoost（您之前的代码已经在使用）
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost未安装，将跳过XGBoost模型")
    XGB_AVAILABLE = False

import warnings

warnings.filterwarnings('ignore')

# ========== 数据加载 ==========
data = pd.read_csv('SAP6_ARDS.csv')
np.set_printoptions(suppress=True)
y = data["ARDS"].values
X = data.iloc[:, 0:32].values

# 计算正类比例，用于处理不平衡数据
pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1.0

# ========== 定义所有要比较的模型（仅使用scikit-learn和XGBoost） ==========
models = {}

# 1. 线性模型
models["Logistic Regression"] = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# 2. 支持向量机
models["SVM (RBF)"] = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

models["Linear SVM"] = LinearSVC(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# 3. 决策树
models["Decision Tree"] = DecisionTreeClassifier(
    class_weight='balanced',
    random_state=42
)

# 4. 集成学习（随机森林家族）
models["Random Forest"] = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

models["Extra Trees"] = ExtraTreesClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# 5. 梯度提升
models["Gradient Boosting"] = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

models["AdaBoost"] = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

# 6. Bagging
models["Bagging (DT)"] = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(class_weight='balanced'),
    n_estimators=50,
    random_state=42
)

# 7. 神经网络
models["MLP (Deep)"] = MLPClassifier(
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

# 8. 贝叶斯
models["Gaussian Naive Bayes"] = GaussianNB()

# 9. 判别分析
models["Linear Discriminant"] = LinearDiscriminantAnalysis()
models["Quadratic Discriminant"] = QuadraticDiscriminantAnalysis()

# 10. 高斯过程
# kernel = 1.0 * RBF(1.0)
# models["Gaussian Process"] = GaussianProcessClassifier(
#     kernel=kernel,
#     random_state=42
# )

# 11. K近邻
models["KNN (k=5)"] = KNeighborsClassifier(n_neighbors=5)
models["KNN (k=10)"] = KNeighborsClassifier(n_neighbors=10)

# 12. XGBoost（如果可用）
if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
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

# 13. 堆叠集成（使用已有的模型）
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

if XGB_AVAILABLE:
    base_estimators.insert(1, (
        'xgb', XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss')))

models["Stacking Ensemble"] = StackingClassifier(
    estimators=base_estimators[:3],  # 只取前3个
    final_estimator=LogisticRegression(),
    cv=3
)

print(f"总共 {len(models)} 个模型将被测试")
print("模型列表:", list(models.keys()))

# ========== 设置交叉验证 ==========
n_folds = 30
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ========== 存储所有模型的结果 ==========
results = {}
all_fold_results = []  # 存储每一折的详细结果

# ========== 为ROC和PR曲线存储数据 ==========
all_roc_data = {}
all_pr_data = {}

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ========== 对每个模型进行训练和评估 ==========
for model_name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"正在训练模型: {model_name} ({list(models.keys()).index(model_name) + 1}/{len(models)})")
    print('=' * 60)

    # 存储当前模型的指标
    model_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': [],
        'tpr': [],  # sensitivity
        'tnr': [],  # specificity
        'ppv': [],  # precision
        'npv': [],  # negative predictive value
        'predictions': [],
        'probabilities': [],
        'fold_indices': []  # 存储每一折的索引
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

        try:
            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 对于某些模型（如LinearSVC），需要特殊处理概率预测
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                # 对于没有predict_proba的模型，使用决策函数
                if hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    # 将决策函数转换为概率（粗略估计）
                    from scipy.special import expit

                    y_pred_proba = expit(decision_scores)
                else:
                    # 如果既没有predict_proba也没有decision_function，跳过概率相关指标
                    y_pred_proba = y_pred.astype(float)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

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
            model_results['tpr'].append(sensitivity)
            model_results['tnr'].append(specificity)
            model_results['ppv'].append(ppv)
            model_results['npv'].append(npv)
            model_results['predictions'].append(y_pred)
            model_results['probabilities'].append(y_pred_proba)
            model_results['fold_indices'].append((train_idx, test_idx))

            # 存储到全局的每一折结果列表
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc,
                'Sensitivity_TPR': sensitivity,
                'Specificity_TNR': specificity,
                'PPV': ppv,
                'NPV': npv,
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

            # 即使失败也记录
            fold_result = {
                'Model': model_name,
                'Fold': fold,
                'Accuracy': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1_Score': np.nan,
                'ROC_AUC': np.nan,
                'PR_AUC': np.nan,
                'Sensitivity_TPR': np.nan,
                'Specificity_TNR': np.nan,
                'PPV': np.nan,
                'NPV': np.nan,
                'Train_Samples': len(train_idx),
                'Test_Samples': len(test_idx),
                'Test_Positive_Rate': np.mean(y_test),
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
            'Mean_Accuracy': np.nan,
            'Std_Accuracy': np.nan,
            'Mean_Precision': np.nan,
            'Std_Precision': np.nan,
            'Mean_Recall': np.nan,
            'Std_Recall': np.nan,
            'Mean_F1': np.nan,
            'Std_F1': np.nan,
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
            'Status': 'FAILED'
        }
        continue

    # 计算平均指标和标准差
    model_results['mean_accuracy'] = np.mean(model_results['accuracy'])
    model_results['mean_precision'] = np.mean(model_results['precision'])
    model_results['mean_recall'] = np.mean(model_results['recall'])
    model_results['mean_f1'] = np.mean(model_results['f1'])
    model_results['mean_roc_auc'] = np.mean(model_results['roc_auc'])
    model_results['mean_pr_auc'] = np.mean(model_results['pr_auc'])
    model_results['mean_tpr'] = np.mean(model_results['tpr'])
    model_results['mean_tnr'] = np.mean(model_results['tnr'])
    model_results['mean_ppv'] = np.mean(model_results['ppv'])
    model_results['mean_npv'] = np.mean(model_results['npv'])

    model_results['std_accuracy'] = np.std(model_results['accuracy'])
    model_results['std_precision'] = np.std(model_results['precision'])
    model_results['std_recall'] = np.std(model_results['recall'])
    model_results['std_f1'] = np.std(model_results['f1'])
    model_results['std_roc_auc'] = np.std(model_results['roc_auc'])
    model_results['std_pr_auc'] = np.std(model_results['pr_auc'])
    model_results['std_tpr'] = np.std(model_results['tpr'])
    model_results['std_tnr'] = np.std(model_results['tnr'])
    model_results['std_ppv'] = np.std(model_results['ppv'])
    model_results['std_npv'] = np.std(model_results['npv'])

    # 存储ROC曲线数据
    all_roc_data[model_name] = {
        'mean_fpr': mean_fpr,
        'mean_tpr': np.mean(tprs, axis=0),
        'std_tpr': np.std(tprs, axis=0),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'tprs': tprs
    }

    # 存储PR曲线数据
    all_pr_data[model_name] = {
        'mean_recall': mean_recall,
        'mean_precision': np.mean(precisions, axis=0),
        'std_precision': np.std(precisions, axis=0),
        'mean_pr_auc': np.mean(pr_aucs),
        'std_pr_auc': np.std(pr_aucs),
        'precisions': precisions
    }

    # 存储当前模型的结果
    results[model_name] = model_results

    # 创建模型汇总结果
    summary_result = {
        'Model': model_name,
        'Total_Folds': n_folds,
        'Successful_Folds': successful_folds,
        'Mean_Accuracy': model_results['mean_accuracy'],
        'Std_Accuracy': model_results['std_accuracy'],
        'Mean_Precision': model_results['mean_precision'],
        'Std_Precision': model_results['std_precision'],
        'Mean_Recall': model_results['mean_recall'],
        'Std_Recall': model_results['std_recall'],
        'Mean_F1': model_results['mean_f1'],
        'Std_F1': model_results['std_f1'],
        'Mean_ROC_AUC': model_results['mean_roc_auc'],
        'Std_ROC_AUC': model_results['std_roc_auc'],
        'Mean_PR_AUC': model_results['mean_pr_auc'],
        'Std_PR_AUC': model_results['std_pr_auc'],
        'Mean_Sensitivity': model_results['mean_tpr'],
        'Std_Sensitivity': model_results['std_tpr'],
        'Mean_Specificity': model_results['mean_tnr'],
        'Std_Specificity': model_results['std_tnr'],
        'Mean_PPV': model_results['mean_ppv'],
        'Std_PPV': model_results['std_ppv'],
        'Mean_NPV': model_results['mean_npv'],
        'Std_NPV': model_results['std_npv'],
        'Status': 'SUCCESS'
    }
    all_fold_results.append(summary_result)  # 也添加到列表中以保持结构

    # 打印当前模型的性能
    print(f"\n{model_name} 性能汇总 ({successful_folds}/{n_folds} folds成功):")
    print(f"  ROC AUC: {model_results['mean_roc_auc']:.4f} ± {model_results['std_roc_auc']:.4f}")
    print(f"  PR AUC:  {model_results['mean_pr_auc']:.4f} ± {model_results['std_pr_auc']:.4f}")
    print(f"  F1 Score: {model_results['mean_f1']:.4f}")
    print(f"  Recall:   {model_results['mean_recall']:.4f}")
    print(f"  Precision: {model_results['mean_precision']:.4f}")

# ========== 保存结果到CSV文件 ==========
print(f"\n{'=' * 60}")
print("正在保存结果到CSV文件...")
print('=' * 60)

# 1. 保存每一折的详细结果
if all_fold_results:
    # 分离fold详细结果和模型汇总结果
    fold_details = [r for r in all_fold_results if 'Fold' in r]
    model_summaries = [r for r in all_fold_results if 'Total_Folds' in r]

    # 保存每一折详细结果
    if fold_details:
        fold_df = pd.DataFrame(fold_details)
        fold_csv_path = os.path.join(results_dir, f'fold_results_detailed.csv')
        fold_df.to_csv(fold_csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存每一折详细结果到: {fold_csv_path}")
        print(f"  共 {len(fold_df)} 条记录，{len(fold_df.columns)} 个指标")

    # 保存模型汇总结果
    if model_summaries:
        summary_df = pd.DataFrame(model_summaries)
        summary_csv_path = os.path.join(results_dir, f'model_summary_results.csv')
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存模型汇总结果到: {summary_csv_path}")
        print(f"  共 {len(summary_df)} 个模型，{len(summary_df.columns)} 个指标")

    # 创建综合结果表格（合并详细和汇总）
    if fold_details and model_summaries:
        # 将fold结果按模型分组，计算额外的统计信息
        fold_grouped = fold_df.groupby('Model')

        # 创建一个更详细的汇总表
        detailed_summary = []
        for model_name, group in fold_grouped:
            if 'Mean_Accuracy' in group.columns:  # 跳过已经是汇总的行
                continue

            # 计算额外的统计量
            successful_folds = group.dropna(subset=['Accuracy']).shape[0]
            coef_variation_roc = group['ROC_AUC'].std() / group['ROC_AUC'].mean() if group[
                                                                                         'ROC_AUC'].mean() != 0 else np.nan
            coef_variation_pr = group['PR_AUC'].std() / group['PR_AUC'].mean() if group[
                                                                                      'PR_AUC'].mean() != 0 else np.nan

            # 找到最优的fold（按F1分数）
            best_fold_idx = group['F1_Score'].idxmax() if not group['F1_Score'].isna().all() else None
            best_fold = group.loc[best_fold_idx] if best_fold_idx is not None else None

            model_summary = {
                'Model': model_name,
                'Total_Folds_Attempted': n_folds,
                'Successful_Folds': successful_folds,
                'Success_Rate': successful_folds / n_folds * 100,
                'Mean_Accuracy': group['Accuracy'].mean(),
                'Std_Accuracy': group['Accuracy'].std(),
                'Mean_Precision': group['Precision'].mean(),
                'Std_Precision': group['Precision'].std(),
                'Mean_Recall': group['Recall'].mean(),
                'Std_Recall': group['Recall'].std(),
                'Mean_F1_Score': group['F1_Score'].mean(),
                'Std_F1_Score': group['F1_Score'].std(),
                'Mean_ROC_AUC': group['ROC_AUC'].mean(),
                'Std_ROC_AUC': group['ROC_AUC'].std(),
                'Mean_PR_AUC': group['PR_AUC'].mean(),
                'Std_PR_AUC': group['PR_AUC'].std(),
                'CV_ROC_AUC': coef_variation_roc * 100 if not np.isnan(coef_variation_roc) else np.nan,
                'CV_PR_AUC': coef_variation_pr * 100 if not np.isnan(coef_variation_pr) else np.nan,
                'Min_ROC_AUC': group['ROC_AUC'].min(),
                'Max_ROC_AUC': group['ROC_AUC'].max(),
                'Min_PR_AUC': group['PR_AUC'].min(),
                'Max_PR_AUC': group['PR_AUC'].max(),
                'Best_Fold_Number': best_fold['Fold'] if best_fold is not None else np.nan,
                'Best_Fold_F1': best_fold['F1_Score'] if best_fold is not None else np.nan,
                'Best_Fold_ROC_AUC': best_fold['ROC_AUC'] if best_fold is not None else np.nan,
                'Best_Fold_PR_AUC': best_fold['PR_AUC'] if best_fold is not None else np.nan,
                'Mean_Sensitivity': group['Sensitivity_TPR'].mean(),
                'Mean_Specificity': group['Specificity_TNR'].mean(),
                'Mean_PPV': group['PPV'].mean(),
                'Mean_NPV': group['NPV'].mean(),
                'Data_Positive_Rate': np.mean(y) * 100,
                'Evaluation_Date': datetime.now().strftime("%Y-%m-%d")
            }
            detailed_summary.append(model_summary)

        if detailed_summary:
            detailed_summary_df = pd.DataFrame(detailed_summary)
            detailed_summary_csv_path = os.path.join(results_dir, f'model_detailed_summary.csv')
            detailed_summary_df.to_csv(detailed_summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"已保存详细模型汇总到: {detailed_summary_csv_path}")

            # 按PR AUC排序并显示前10个模型
            print(f"\n{'=' * 60}")
            print("模型性能排名 (按PR AUC):")
            print('=' * 60)

            ranked_df = detailed_summary_df.sort_values('Mean_PR_AUC', ascending=False)
            display_cols = ['Model', 'Mean_PR_AUC', 'Mean_ROC_AUC', 'Mean_F1_Score',
                            'Mean_Recall', 'Mean_Precision', 'Successful_Folds']
            print(ranked_df[display_cols].head(10).to_string(index=False))

            # 保存排名结果
            rank_csv_path = os.path.join(results_dir, f'model_ranking.csv')
            ranked_df.to_csv(rank_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n已保存模型排名到: {rank_csv_path}")

# ========== 新增：绘制所有模型的ROC曲线 ==========
if len(results) > 0:
    # 选择表现最好的几个模型进行ROC曲线绘制
    top_models_roc = sorted(results.items(),
                            key=lambda x: x[1]['mean_roc_auc'],
                            reverse=True)[:6]

    plt.figure(figsize=(10, 8))

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
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison (Top 6 Models)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, 'ROC_Curves_Comparison.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存ROC曲线图到: {roc_plot_path}")

# ========== 新增：绘制所有模型的PR曲线 ==========
if len(results) > 0:
    top_models_pr = sorted(results.items(),
                           key=lambda x: x[1]['mean_pr_auc'],
                           reverse=True)[:6]

    plt.figure(figsize=(10, 8))

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
    plt.title('Precision-Recall Curves Comparison (Top 6 Models)', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(results_dir, 'PR_Curves_Comparison.png')
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存PR曲线图到: {pr_plot_path}")

# ========== 创建综合比较图表 ==========
if len(results) > 0:
    top_models = sorted(results.items(),
                        key=lambda x: x[1]['mean_pr_auc'],
                        reverse=True)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 1. 模型排名柱状图（按PR AUC）
    ax1 = axes[0]
    model_names = [name for name, _ in top_models]
    pr_auc_values = [r['mean_pr_auc'] for _, r in top_models]
    roc_auc_values = [r['mean_roc_auc'] for _, r in top_models]

    x_pos = np.arange(len(model_names))
    width = 0.35

    ax1.bar(x_pos - width / 2, pr_auc_values, width, label='PR AUC', alpha=0.8, color='skyblue')
    ax1.bar(x_pos + width / 2, roc_auc_values, width, label='ROC AUC', alpha=0.8, color='lightcoral')

    ax1.set_xlabel('Models')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Top Model Ranking by PR AUC')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (pr, roc) in enumerate(zip(pr_auc_values, roc_auc_values)):
        ax1.text(i - width / 2, pr + 0.01, f'{pr:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width / 2, roc + 0.01, f'{roc:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. F1-Recall-Precision雷达图
    ax2 = axes[1]
    metrics = ['F1', 'Recall', 'Precision', 'Accuracy']
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()

    for model_name, model_results in top_models[:4]:
        values = [
            model_results['mean_f1'],
            model_results['mean_recall'],
            model_results['mean_precision'],
            model_results['mean_accuracy']
        ]

        values += values[:1]
        current_angles = angles + angles[:1]

        ax2.plot(current_angles, values, 'o-', linewidth=2, label=model_name)
        ax2.fill(current_angles, values, alpha=0.1)

    ax2.set_xticks(angles)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Radar Chart (Top 4 Models)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True)

    # 3. 性能热力图
    ax3 = axes[2]
    performance_matrix = np.array([
        [r['mean_roc_auc'] for _, r in top_models],
        [r['mean_pr_auc'] for _, r in top_models],
        [r['mean_f1'] for _, r in top_models],
        [r['mean_recall'] for _, r in top_models],
        [r['mean_precision'] for _, r in top_models]
    ])

    im = ax3.imshow(performance_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    ax3.set_xticks(np.arange(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax3.set_yticks(np.arange(5))
    ax3.set_yticklabels(['ROC AUC', 'PR AUC', 'F1', 'Recall', 'Precision'])
    ax3.set_title('')

    # 添加数值标签
    for i in range(performance_matrix.shape[0]):
        for j in range(performance_matrix.shape[1]):
            text = ax3.text(j, i, f'{performance_matrix[i, j]:.3f}',
                            ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax3)

    # 4. 散点图：Recall vs Precision
    ax4 = axes[3]
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_models)))

    for idx, (model_name, model_results) in enumerate(top_models):
        recall = model_results['mean_recall']
        precision = model_results['mean_precision']
        f1 = model_results['mean_f1']

        ax4.scatter(recall, precision, s=f1 * 200, color=colors[idx],
                    alpha=0.7, label=f'{model_name} (F1={f1:.3f})')
        ax4.annotate(model_name[:15], (recall, precision),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Recall vs Precision (Bubble size = F1 Score)')
    ax4.set_xlim(0, 1.05)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=8)

    # 绘制随机基线
    pos_rate = np.mean(y)
    ax4.axhline(y=pos_rate, color='gray', linestyle='--', alpha=0.5, label=f'Random Precision ({pos_rate:.3f})')
    ax4.axvline(x=pos_rate, color='gray', linestyle='--', alpha=0.5, label=f'Random Recall ({pos_rate:.3f})')

    plt.tight_layout()
    comparison_plot_path = os.path.join(results_dir, 'Model_Comparison_Top4.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存综合比较图到: {comparison_plot_path}")

    # ========== 打印详细性能对比表格 ==========
    print("\n" + "=" * 100)
    print("TOP MODEL PERFORMANCE COMPARISON TABLE (按PR AUC排序)")
    print("=" * 100)

    # 准备表格数据（所有模型，按PR AUC排序）
    all_sorted_models = sorted(results.items(),
                               key=lambda x: x[1]['mean_pr_auc'],
                               reverse=True)

    table_data = []
    for model_name, r in all_sorted_models:
        table_data.append([
            model_name,
            f"{r['mean_pr_auc']:.4f}",
            f"{r['mean_roc_auc']:.4f}",
            f"{r['mean_f1']:.4f}",
            f"{r['mean_recall']:.4f}",
            f"{r['mean_precision']:.4f}",
            f"{r['mean_accuracy']:.4f}",
            f"{(r['mean_tpr'] + r['mean_tnr']) / 2:.4f}"
        ])

    # 创建并显示表格
    headers = ["Model", "PR AUC", "ROC AUC", "F1", "Recall", "Precision", "Accuracy", "Avg(TPR+TNR)"]
    print("\n" + " | ".join([f"{h:^20}" for h in headers]))
    print("-" * 180)

    for row in table_data:
        print(" | ".join([f"{str(item):^20}" for item in row]))

    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)

    # 找出最佳模型
    best_model_name, best_model_results = all_sorted_models[0]

    print(f"1. 最佳模型 (按PR AUC): {best_model_name}")
    print(f"   - PR AUC: {best_model_results['mean_pr_auc']:.4f}")
    print(f"   - ROC AUC: {best_model_results['mean_roc_auc']:.4f}")
    print(f"   - F1 Score: {best_model_results['mean_f1']:.4f}")
    print(f"   - 召回率(Recall): {best_model_results['mean_recall']:.4f}")
    print(f"   - 精确率(Precision): {best_model_results['mean_precision']:.4f}")

    print(f"\n2. 数据集信息:")
    print(f"   - 总样本数: {len(y)}")
    print(f"   - 正类(SAP)比例: {np.mean(y):.4f} ({sum(y)}/{len(y)})")
    print(f"   - 随机模型PR AUC基线: {np.mean(y):.4f}")

    print(f"\n3. 临床推荐:")
    if best_model_results['mean_recall'] > 0.8:
        print("   ✓ 模型具有高召回率 - 适合用于筛查（漏诊风险低）")
    elif best_model_results['mean_recall'] > 0.6:
        print("   ⚠ 模型召回率中等 - 可考虑调整阈值以提高召回率")
    else:
        print("   ✗ 模型召回率较低 - 可能漏诊较多患者，需要改进")

    if best_model_results['mean_precision'] > 0.7:
        print("   ✓ 模型具有高精确率 - 适合临床决策支持（误诊率低）")
    elif best_model_results['mean_precision'] > 0.5:
        print("   ⚠ 模型精确率中等 - 可能产生一定假阳性")
    else:
        print("   ✗ 模型精确率较低 - 假阳性较多，需谨慎使用")

    # 提供阈值调整建议
    print(f"\n4. 阈值调整建议:")
    if best_model_results['mean_recall'] > best_model_results['mean_precision']:
        print("   - 当前模型偏向高召回率（减少漏诊）")
        print("   - 如需更高精确率（减少误诊），可提高分类阈值")
    elif best_model_results['mean_precision'] > best_model_results['mean_recall']:
        print("   - 当前模型偏向高精确率（减少误诊）")
        print("   - 如需更高召回率（减少漏诊），可降低分类阈值")
    else:
        print("   - 召回率和精确率相对平衡")

    print("=" * 100)

else:
    print("没有模型成功训练，请检查数据或模型配置。")

print(f"\n{'=' * 60}")
print("所有结果已保存完毕！")
print(f"结果保存在目录: {results_dir}")
print('=' * 60)

# 显示保存的文件列表
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f"\n目录中包含以下文件:")
    for file in sorted(files):
        file_path = os.path.join(results_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # 转换为KB
        print(f"  - {file} ({file_size:.1f} KB)")