import json

import mlflow
import mlflow.catboost


def load_base_categories():
    with open("data/ru_ecomm_tree_category.json", encoding="utf-8") as file:
        base_categories = json.load(file)
    return base_categories


def load_models():
    cl_model_cat_l1 = mlflow.catboost.load_model(
        "models:/CLASSIFIER_MODEL_CAT_L1/Production"
    )
    cl_model_cat_l2 = mlflow.catboost.load_model(
        "models:/CLASSIFIER_MODEL_CAT_L2/Production"
    )
    cl_model_cat_l3 = mlflow.catboost.load_model(
        "models:/CLASSIFIER_MODEL_CAT_L3/Production"
    )

    cat_by_model_l1 = {
        idx: cat_name for idx, cat_name in enumerate(cl_model_cat_l1.classes_)
    }
    cat_by_model_l2 = {
        idx: cat_name for idx, cat_name in enumerate(cl_model_cat_l2.classes_)
    }
    cat_by_model_l3 = {
        idx: cat_name for idx, cat_name in enumerate(cl_model_cat_l3.classes_)
    }

    return (
        cl_model_cat_l1,
        cl_model_cat_l2,
        cl_model_cat_l3,
        cat_by_model_l1,
        cat_by_model_l2,
        cat_by_model_l3
    )
