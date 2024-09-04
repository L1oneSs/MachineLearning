import numpy as np
import matplotlib.pyplot as plt

n = 1000

# Параметры для футболистов
mu_football = 160
sigma_football = 10

# Параметры для баскетболистов
mu_basketball = 180
sigma_basketball = 10

# Генерация роста для футболистов и баскетболистов
football_heights = np.random.normal(mu_football, sigma_football, n)
basketball_heights = np.random.normal(mu_basketball, sigma_basketball, n)


def binary_classifier(heights, threshold):
    predictions = [1 if height >= threshold else 0 for height in heights]
    return predictions


def calculate_metrics(true_labels, predicted_labels):
    TP = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1])
    TN = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 0])
    FP = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1])
    FN = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0])

    accuracy = (TP + TN) / len(true_labels)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    alpha = FP / (FP + TN) if FP + TN > 0 else 0
    beta = FN / (FN + TP) if FN + TP > 0 else 0

    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
        'Error Type I (alpha)': alpha,
        'Error Type II (beta)': beta,
        'FPR': FPR,
        'TPR': TPR
    }
    return metrics


# Генерация данных для классификации
football_labels = [0] * n
basketball_labels = [1] * n

all_heights = np.concatenate([football_heights, basketball_heights])
all_labels = np.concatenate([football_labels, basketball_labels])

# Порог
T = 169

predictions = binary_classifier(all_heights, T)

metrics = calculate_metrics(all_labels, predictions)

print("Метрики классификации:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# <================================================================================>

tpr_values = []
fpr_values = []
metrics_values = []

for threshold in range(T):
    # Применяем классификатор с текущим порогом
    predictions = binary_classifier(all_heights, threshold)

    metrics = calculate_metrics(all_labels, predictions)

    metrics_values.append((metrics, threshold))

    tpr_values.append(metrics['TPR'])
    fpr_values.append(metrics['FPR'])

# Разделим массив all_labels на два массива для футболистов и баскетболистов
football_labels = all_labels[:n]
basketball_labels = all_labels[n:]

# Строим ROC-кривую
plt.figure()
plt.plot(fpr_values, tpr_values, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Вычисляем AUC
sorted_indices = np.argsort(fpr_values)
tpr_values = np.array(tpr_values)[sorted_indices]
fpr_values = np.array(fpr_values)[sorted_indices]

roc_auc = np.trapz(tpr_values, fpr_values)

roc_auc_manual = 0
for i in range(1, len(fpr_values)):
    roc_auc_manual += 0.5 * (tpr_values[i] + tpr_values[i - 1]) * (fpr_values[i] - fpr_values[i - 1])

print("\nAUC (Area Under Curve) <Auto>:", roc_auc)
print("AUC (Area Under Curve): <Hands>", roc_auc_manual)

# Находим порог, при котором достигается максимальное значение Accuracy
best_metrics, best_threshold = max(metrics_values, key=lambda x: x[0]['Accuracy'])

# Выводим метрики для порога с максимальным Accuracy
print(f"\nПорог с максимальным Accuracy: {best_threshold}")
print("\nМетрики для порога с максимальным Accuracy:")
for metric, value in best_metrics.items():
    print(f"{metric}: {value}")


print("\n<================================================>\n")
# <==========================================================>
print("\nМетод 2\n")
# Переменные для хранения TPR и FPR
tpr_values_manual = []
fpr_values_manual = []
auc_manual = 0

# Сортируем предсказанные значения в порядке убывания
predicted_sorted_indices = np.argsort(all_heights)[::-1]
sorted_predictions = [all_labels[i] for i in predicted_sorted_indices]

# Инициализируем счетчики TP, FP, TN, FN
TP = 0
FP = 0
TN = sorted_predictions.count(0)
FN = sorted_predictions.count(1)

# Переменные для хранения метрик для лучшего порога
best_threshold_metrics = None
best_threshold = None
best_accuracy = 0

# Проходим по отсортированным предсказанным значениям
for label in sorted_predictions:
    # Обновляем счетчики в зависимости от предсказанного класса
    if label == 1:
        TP += 1
        FN -= 1
    else:
        FP += 1
        TN -= 1

    # Вычисляем метрики
    accuracy = (TP + TN) / len(all_labels)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Обновляем лучший порог, если Accuracy увеличивается
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold_metrics = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1_score
        }
        best_threshold = all_heights[predicted_sorted_indices]

    # Вычисляем TPR и FPR
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    # Добавляем TPR и FPR в соответствующие списки
    tpr_values_manual.append(tpr)
    fpr_values_manual.append(fpr)

    # Вычисляем AUC
    if len(tpr_values_manual) > 1:
        auc_manual += (fpr_values_manual[-1] - fpr_values_manual[-2]) * tpr_values_manual[-1]

'''
# Выводим метрики для лучшего порога
print(f"\nЛучший порог с максимальным Accuracy: {best_threshold[0]}")
print("\nМетрики для лучшего порога:")
for metric, value in best_threshold_metrics.items():
    print(f"{metric}: {value}")
'''
# Строим ROC-кривую
plt.figure()
plt.plot(fpr_values_manual, tpr_values_manual, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Выводим AUC на консоль
print("\nAUC:", auc_manual)





