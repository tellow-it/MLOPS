class ModelRegistry:
    def __init__(self):
        self.cat_model_l1 = None
        self.cat_model_l2 = None
        self.cat_model_l3 = None
        self.categories_l1 = {}
        self.categories_l2 = {}
        self.categories_l3 = {}
        self.base_categories = {}

    def status_load_models(self):
        return self.cat_model_l1 and self.cat_model_l2 and self.cat_model_l3


registry = ModelRegistry()
