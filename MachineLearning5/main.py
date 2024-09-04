import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 entropy_threshold=0.1, parallel_split=True, basis=False):
        self.criterion = criterion
        self.max_depth = max_depth
        # Используется для ограничения:
        # количество элементов обучающей выборки, достигших узла меньше определенного порога
        self.min_samples_split = min_samples_split
        # Минимально необходимое количество записей в подузле, чтобы он не делился и оставался листовым
        self.min_samples_leaf = min_samples_leaf
        # Порог энтропии. Если энтропия < порогового значения, разделение прекращается и узел
        # остается листовым
        self.entropy_threshold = entropy_threshold
        self.tree = None
        self.parallel_split = parallel_split
        self.basis = basis

    def information_gain(self, parent, left_child, right_child):
        """
        :param parent: Родительский узел до разделения
        :param left_child: Левый узел после разделения
        :param right_child: Правый узел после разделения
        :return: Информационный выигрыш(Прирост информации)
        """
        # Доля данных, попавших в левый дочерний узел после разделения
        num_left = len(left_child) / len(parent)
        # Доля данных, попавших в правый дочерний узел после разделения
        num_right = len(right_child) / len(parent)
        if self.criterion == 'entropy':
            ig = self.entropy(parent) - (num_left * self.entropy(left_child) + num_right * self.entropy(right_child))
        elif self.criterion == 'gini':
            ig = self.gini(parent) - (num_left * self.gini(left_child) + num_right * self.gini(right_child))
        elif self.criterion == 'misclassification':
            ig = self.misclassification_error(parent) - (
                        num_left * self.misclassification_error(left_child) + num_right * self.misclassification_error(
                    right_child))
        else:
            raise ValueError("Error")
        return ig

    def entropy(self, y):
        # Вычисляем уникальные классы в целевой переменной и количество их появлений
        classes, counts = np.unique(y, return_counts=True)
        # Вычисляем вероятность для каждого класса
        probabilities = counts / len(y)
        # Вычисляем энтропию по формуле Шеннона
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    # Неопределенность Джини
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        gini_index = 1 - np.sum((counts / len(y)) ** 2)
        return gini_index

    # Ошибка классификации
    def misclassification_error(self, y):
        classes, counts = np.unique(y, return_counts=True)
        if len(counts) == 0:
            return 0
        max_probability = np.max(counts / len(y))
        error = 1 - max_probability
        return error

    def split(self, X, y, feature_index, threshold):
        """
        :param X: Матрица признаков
        :param y: Целевая переменная
        :param feature_index: Номер индекса, по которому хотим произвести разделение
        :param threshold: Пороговое значение, по которому данные будут делиться
        :return: 4 массива с разделенными данными
        """
        # Влево идут данные, для которых значение признака <= порогу
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        # Вправо идут данные, для которых значение признака > порога
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]
        return left_X, left_y, right_X, right_y

    # Метод поиска наилучшего разделения на два подмножества
    def split_best(self, X, y):
        """
        :return: Наилучший признак и порог для разделения данных
        """
        best_feature_index = None
        best_threshold = None
        best_ig = -np.inf

        if not self.parallel_split:
            # Непараллельное разделение

            # Генерируем два случайных индекса признаков
            feature_indices = np.random.choice(X.shape[1], size=2, replace=False)
            # Извлекаем случайные признаки из данных
            feature1, feature2 = X[:, feature_indices[0]], X[:, feature_indices[1]]
            # Вычисляем пространственные различия, чтобы понять, как сильно два признака
            # различаются между собой
            margins = np.abs(feature2) * np.linalg.norm(feature1) - np.abs(feature1) * np.linalg.norm(feature2)
            # Выбор индекса признака с наибольшим пространственным различием
            best_index = np.argmax(margins)
            # Лучше всего разделять по данным признакам с усредненным порогом
            best_feature_index = (feature_indices[0], feature_indices[1])
            best_threshold = (X[best_index, feature_indices[0]] + X[best_index, feature_indices[1]]) / 2.0


        elif self.basis:

            # Разделение базисными функциями

            #poly = PolynomialFeatures(degree=2, include_bias=False)

            #poly_X = poly.fit_transform(X)

            sin_X = np.sin(X)

            cos_X = np.cos(X)

            new_features = sin_X

            for feature_index in range(new_features.shape[1]):
                thresholds = np.unique(new_features[:, feature_index])

                for threshold in thresholds:

                    left_X, left_y, right_X, right_y = self.split(new_features, y, feature_index, threshold)

                    ig = self.information_gain(y, left_y, right_y)

                    if ig > best_ig:
                        best_ig = ig

                        best_feature_index = feature_index

                        best_threshold = threshold


        elif self.parallel_split:
            # Параллельное разделение
            # Проход по всем признакам
            for feature_index in range(X.shape[1]):
                # Проход по всем уникальным порогам
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    # Разделение данных
                    left_X, left_y, right_X, right_y = self.split(X, y, feature_index, threshold)
                    # Вычисление информационного выигрыша и сравнение
                    ig = self.information_gain(y, left_y, right_y)
                    if ig > best_ig:
                        best_ig = ig
                        best_feature_index = feature_index
                        best_threshold = threshold

        return best_feature_index, best_threshold

    # Метод построения дерева
    def build_tree(self, X, y, depth=0):
        # Проверка условий остановки
        if (self.max_depth is not None and depth >= self.max_depth) \
                or len(X) < self.min_samples_split \
                or self.entropy(y) < self.entropy_threshold:
            # Возвращаем метку класса с наибольшим количеством образцов
            return np.argmax(np.bincount(y))

        # Ищем наилучшие параметры разделения для текущего узла
        best_feature_index, best_threshold = self.split_best(X, y)
        if best_feature_index is None:
            # Возвращаем метку класса с наибольшим количеством образцов
            return np.argmax(np.bincount(y))

        # Разделяем текущий узел
        left_X, left_y, right_X, right_y = self.split(X, y, best_feature_index, best_threshold)

        # Проверка условий для создания терминального узла
        if len(left_y) < self.min_samples_leaf \
                or len(right_y) < self.min_samples_leaf:
            return np.argmax(np.bincount(y))

        # Рекурсия для создания левого и правого поддерева
        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        right_subtree = self.build_tree(right_X, right_y, depth + 1)

        # Возвращаем текущий узел и его поддеревья
        return {'feature_index': best_feature_index, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    # Метод обучения дерева решений
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if isinstance(tree, (int, np.int64)):
            return tree  # если узел является целым числом, возвращаем его как терминальный узел
        feature_index, threshold = tree.get('feature_index'), tree.get('threshold')
        if feature_index is None or threshold is None:
            return tree  # если узел не имеет признака и порога, возвращаем его как терминальный узел

        if isinstance(feature_index, tuple):  # Для непараллельного разделения
            if isinstance(threshold, tuple):  # Если threshold является кортежем, используем индексацию
                if x[feature_index[0]] <= threshold[0] and x[feature_index[1]] <= threshold[1]:
                    return self.predict_sample(x, tree['left'])
                else:
                    return self.predict_sample(x, tree['right'])
            else:  # Иначе threshold является скалярным значением
                if x[feature_index[0]] <= threshold and x[feature_index[1]] <= threshold:
                    return self.predict_sample(x, tree['left'])
                else:
                    return self.predict_sample(x, tree['right'])
        elif isinstance(feature_index, int):  # Для параллельного разделения
            if isinstance(threshold, tuple):  # Если threshold является кортежем
                if x[feature_index] <= threshold[0] and x[feature_index] <= threshold[1]:
                    return self.predict_sample(x, tree['left'])
                else:
                    return self.predict_sample(x, tree['right'])
            else:  # Иначе threshold является скалярным значением
                if x[feature_index] <= threshold:
                    return self.predict_sample(x, tree['left'])
                else:
                    return self.predict_sample(x, tree['right'])

    # Метод прогнозирования метки класса X с использованием обученного дерева решений
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.tree))
        return np.array(predictions)


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='entropy', max_depth=10, entropy_threshold=0.05,
                           parralel=True, basis=False):
    # Создание и обучение модели Decision Tree
    clf = DecisionTree(criterion=criterion, max_depth=max_depth, entropy_threshold=entropy_threshold,
                       parallel_split=parralel, basis=basis)
    clf.fit(X_train, y_train)

    # Предсказание на тестовом наборе данных
    y_pred = clf.predict(X_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy sklearn:", accuracy)

    # Подсчет количества правильных предсказаний
    correct_predictions = np.sum(y_pred == y_test)
    # Вычисление общего количества предсказаний
    total_predictions = len(y_test)
    # Вычисление точности модели как доли правильных предсказаний
    accuracy = correct_predictions / total_predictions
    print("Accuracy handmade:", accuracy)

    # Построение confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix sklearn:")
    print(conf_matrix)

    # Инициализация confusion matrix
    num_classes = len(np.unique(y_test))
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    # Заполнение confusion matrix
    for true_label, predicted_label in zip(y_test, y_pred):
        conf_matrix[true_label, predicted_label] += 1
    print("Confusion Matrix handmade:")
    print(conf_matrix)

    from collections import Counter

    # Создание словаря для подсчета количества правильно предсказанных классов
    class_counts = Counter()

    # Проход по всем элементам тестового набора данных
    for true_label, predicted_label in zip(y_test, y_pred):
        # Если предсказанная метка совпадает с истинной меткой, увеличиваем счетчик для этого класса
        if true_label == predicted_label:
            class_counts[true_label] += 1

    # Получение уникальных меток классов и их количество
    unique_classes, counts = zip(*class_counts.items())

    # Построение гистограммы
    plt.bar(unique_classes, counts)
    plt.title("Correct Predictions per Class")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency of Correct Predictions")
    plt.show()

    # Создание словаря для подсчета количества правильно предсказанных классов
    class_counts = Counter()

    # Проход по всем элементам тестового набора данных
    for true_label, predicted_label in zip(y_test, y_pred):
        # Если предсказанная метка совпадает с истинной меткой, увеличиваем счетчик для этого класса
        if true_label != predicted_label:
            class_counts[true_label] += 1

    # Получение уникальных меток классов и их количество
    unique_classes, counts = zip(*class_counts.items())

    # Построение гистограммы
    plt.bar(unique_classes, counts)
    plt.title("Incorrect Predictions per Class")
    plt.xlabel("Class Label")
    plt.ylabel("Frequency of Incorrect Predictions")
    plt.show()


# <=================================Основное задание==================================>
digits = load_digits()
X, y = digits.data, digits.target

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='entropy', max_depth=10, entropy_threshold=0.05)

# <=================================ДОП 1==================================>
print(f'\n<===============ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ 1================>\n')
evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='gini', max_depth=10, entropy_threshold=0.05,
                       parralel=True, basis=False)

# <=================================ДОП 2==================================>
print(f'\n<===============ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ 2================>\n')
evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='misclassification', max_depth=10,
                       entropy_threshold=0.05, parralel=True, basis=False)

# <=================================ДОП 3==================================>
print(f'\n<===============ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ 3================>\n')
# Загрузка датасета
digits = load_digits()
X, y = digits.data, digits.target

# Разделение на тренировочный, валидационный и тестовый наборы данных
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)

# Определение диапазонов значений гиперпараметров
max_depth_values = [10, 20, 30]
criterion_values = ['entropy', 'gini', 'misclassification']
entropy_threshold_values = [0.05, 0.1, 0.15]
min_samples_leaf_values = [1, 2, 4, 8]

# Вложенные циклы для перебора всех комбинаций гиперпараметров
epoch = 0
best_accuracy = 0
best_params = {}

for max_depth in max_depth_values:
    for criterion in criterion_values:
        for entropy_threshold in entropy_threshold_values:
            for min_samples_leaf in min_samples_leaf_values:
                epoch += 1
                print(f"Epoch {epoch}/{len(max_depth_values) * len(criterion_values) * len(entropy_threshold_values) * len(min_samples_leaf_values)}")
                print(f"Max Depth: {max_depth}, Criterion: {criterion}, Entropy Threshold: {entropy_threshold}, Min Samples Leaf: {min_samples_leaf}")

                # Создание модели с текущими гиперпараметрами
                clf = DecisionTree(criterion=criterion, max_depth=max_depth, entropy_threshold=entropy_threshold,
                                   min_samples_leaf=min_samples_leaf, parallel_split=True)

                # Обучение модели
                clf.fit(X_train, y_train)

                # Оценка точности на валидационном наборе данных
                y_val_pred = clf.predict(X_val)
                accuracy = accuracy_score(y_val, y_val_pred)
                print(f"Accuracy on Validation Set: {accuracy}")

                # Проверка, является ли текущая комбинация гиперпараметров лучшей
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'max_depth': max_depth, 'criterion': criterion,
                                   'entropy_threshold': entropy_threshold, 'min_samples_leaf': min_samples_leaf}
                    print("New best parameters found!")
                    print("Best Parameters:", best_params)

# Обучение модели с лучшими гиперпараметрами на всем тренировочном наборе данных
best_clf = DecisionTree(**best_params)
best_clf.fit(X_train, y_train)

# Оценка производительности модели на тестовом наборе данных
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)


# <=================================ДОП 5==================================>
print(f'\n<===============ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ 5================>\n')
evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='entropy', max_depth=10,
                       entropy_threshold=0.05, parralel=False, basis=False)

# <=================================ДОП 6==================================>
print(f'\n<===============ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ 6================>\n')
evaluate_decision_tree(X_train, X_test, y_train, y_test, criterion='entropy', max_depth=10,
                       entropy_threshold=0.05, parralel=True, basis=True)
