import re
from urllib.parse import urlparse

from slugify import slugify


def extract_keywords(link: str) -> list[str]:
    cleaned_text = re.sub(r"\d+", "", urlparse(link).path)
    clean_text = re.sub(r"[^\w\s]", " ", cleaned_text)
    clean_text = clean_text.replace("_", " ")
    clean_text = clean_text.lower()
    keywords = clean_text.strip().split()
    keywords = [word for word in keywords if word]
    return keywords


def keywords_cleaner(keywords: list[str]) -> list[str]:
    stopwords = [
        "catalog", "item", "product",
        "products", "cat", "dlya", "katalog", "html"
    ]
    keywords = [keyword for keyword in keywords if str(keyword).isalpha()]
    keywords = [keyword for keyword in keywords if len(keyword) > 2]
    keywords = [keyword for keyword in keywords if keyword not in stopwords]
    return keywords


def flatten_categories(categories: dict, parent_category: str = None) -> list[str]:
    result = []
    for category, subcategories in categories.items():
        full_category = f"{parent_category.lower()} {category.lower()}" \
            if parent_category else category

        if isinstance(subcategories, dict):
            result.extend(flatten_categories(subcategories, full_category))
        elif isinstance(subcategories, list) and subcategories:
            for subcategory in subcategories:
                result.append(f"{full_category} {subcategory.lower()}")
        else:
            result.append(full_category)

    return result


def slugify_categories(categories: list[str]) -> list[list[str]]:
    slug_categories = [slugify(category, separator=" ") for category in categories]
    clear_slug_categories = []

    for slug_category in slug_categories:
        slug_category_words = [
            word for word in slug_category.split(" ") if len(word) > 2
        ]
        clear_slug_categories.append(slug_category_words)

    return clear_slug_categories
