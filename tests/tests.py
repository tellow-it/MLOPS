from dataclasses import FrozenInstanceError

import pytest

from ml_product_categorizer.scripts.preprocessing import (
    clear_product_text,
    extract_keywords,
)
from src.params import Model


@pytest.mark.parametrize("link,expected", [
    (
        "https://example.com/product/1234_super-widget_v2",
        ["product", "super", "widget", "v"]
    ),
    (
        "https://shop.site.com/phones/samsung-galaxy_s22-ultra",
        ["phones", "samsung", "galaxy", "s", "ultra"]
    ),
    ("https://abc.com/abc_def/123", ["abc", "def"]),
    ("https://abc.com/", []),
    ("https://abc.com/123/456", []),
])
def test_extract_keywords(link, expected):
    assert extract_keywords(link) == expected


@pytest.mark.parametrize("input_text,expected", [
    ("apple 123 orange short 99 banana", "apple orange short banana"),
    ("1 23 456 12", ""),
    ("good product longword and 4567", "good product longword"),
    ("", ""),
])
def test_clear_product_text(input_text, expected):
    assert clear_product_text(input_text) == expected


def test_model_default_values():
    model = Model()
    assert model.random_state == 777
    assert model.n_estimators == 200
    assert model.max_depth == 8
    assert model.max_features == 5


def test_model_is_frozen():
    model = Model()
    with pytest.raises(FrozenInstanceError):
        model.n_estimators = 300
