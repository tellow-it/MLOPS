import numpy as np
from src.ml.model_registry import registry


def model_predict(x: np.array):
    y_proba_l1 = registry.cat_model_l1.predict_proba(x)
    y_cat_l1_idx = np.argmax(y_proba_l1)
    y_cat_l1 = registry.categories_l1[y_cat_l1_idx]

    # проверка есть ли эта категория в списке
    if y_cat_l1 not in registry.base_categories:
        return None

    # проверка есть ли у этой категории l2 уровень
    if len(registry.base_categories[y_cat_l1]) == 0:
        return y_cat_l1

    y_cat_l1_total = y_cat_l1

    x_l2 = np.concatenate([x, y_proba_l1], axis=0)
    y_proba_l2 = registry.cat_model_l2.predict_proba(x_l2)
    y_cat_l2_idx = np.argmax(y_proba_l2)
    y_cat_l2 = registry.categories_l2[y_cat_l2_idx]
    y_cat_l2_part_l1 = y_cat_l2.split(" -> ")[0]
    y_cat_l2_part_l2 = y_cat_l2.split(" -> ")[1]

    # проверка совпал ли l1 уровень при предсказании l2
    if y_cat_l1_total != y_cat_l2_part_l1:
        return y_cat_l1_total

    # проверка есть ли такой l2
    if y_cat_l2_part_l2 not in registry.base_categories[y_cat_l2_part_l1]:
        return y_cat_l1_total

    # проверка есть ли у этой категории l3
    if len(registry.base_categories[y_cat_l2_part_l1][y_cat_l2_part_l2]) == 0:
        return y_cat_l2

    x_l3 = np.concatenate([x_l2, y_proba_l2], axis=0)
    y_proba_l3 = registry.cat_model_l3.predict_proba(x_l3)
    y_cat_l3_idx = np.argmax(y_proba_l3)
    y_cat_l3 = registry.categories_l3[y_cat_l3_idx]

    y_cat_l3_part_l1 = y_cat_l3.split(" -> ")[0]
    y_cat_l3_part_l2 = y_cat_l3.split(" -> ")[1]
    y_cat_l3_part_l3 = y_cat_l3.split(" -> ")[2]

    # проверка есть ли такой l3
    if y_cat_l3_part_l3 not in registry.base_categories[y_cat_l3_part_l1][y_cat_l3_part_l2]:
        return f"{y_cat_l3_part_l1} -> {y_cat_l3_part_l2}"

    return y_cat_l3
