import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    # Сначала создадим массив порогов как среднее между соседними значениями признака
    thresholds = (feature_vector[:-1] + feature_vector[1:]) / 2
    
    # Инициализируем массив для значений критерия Джини для каждого порога
    ginis = np.zeros(len(thresholds))
    
    # Определяем общее количество объектов
    total_samples = len(target_vector)
    
    for i, threshold in enumerate(thresholds):
        # Разбиваем выборку на левое и правое поддерево
        left_mask = feature_vector <= threshold
        right_mask = feature_vector > threshold
        
        left_size = left_mask.sum()
        right_size = right_mask.sum()
        
        # Пропускаем случаи, когда одно из поддеревьев пусто
        if left_size == 0 or right_size == 0:
            continue
        
        # Вычисляем доли объектов классов 0 и 1 в левом и правом поддеревьях
        p0_left = (target_vector[left_mask] == 0).sum() / left_size
        p1_left = 1 - p0_left
        p0_right = (target_vector[right_mask] == 0).sum() / right_size
        p1_right = 1 - p0_right
        
        # Вычисляем значения H(R_l) и H(R_r)
        h_left = 1 - p0_left**2 - p1_left**2
        h_right = 1 - p0_right**2 - p1_right**2
        
        # Вычисляем значение критерия Джини для данного порога
        gini = -left_size / total_samples * h_left - right_size / total_samples * h_right
        ginis[i] = gini
    
    # Находим индекс порога, при котором критерий Джини максимален
    best_index = np.argmax(ginis)
    
    # Оптимальный порог и значение критерия Джини для него
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if len(set(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            if feature_type == "categorical":
                unique_values = np.unique(sub_X[:, feature])
                thresholds = unique_values[:-1] + 0.5  # Use midpoints as thresholds for categorical features
            else:
                feature_vector = sub_X[:, feature]
                thresholds, _, _, _ = find_best_split(feature_vector, sub_y)

            for threshold in thresholds:
                if feature_type == "categorical":
                    split = sub_X[:, feature] == threshold
                else:
                    split = sub_X[:, feature] < threshold

                if np.any(split) and np.any(~split):
                    gini = compute_gini(sub_y[split], sub_y[~split])
                    if gini_best is None or gini < gini_best:
                        feature_best = feature
                        threshold_best = threshold
                        gini_best = gini
                        split_best = split.copy()

        if gini_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
        else:
            node["type"] = "nonterminal"
            node["feature_index"] = feature_best
            node["threshold"] = threshold_best
            node["left_child"], node["right_child"] = {}, {}
            self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"])
            self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
