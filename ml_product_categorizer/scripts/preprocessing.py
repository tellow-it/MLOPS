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


def pipeline_extract_data_4_url(url: str):
    keywords = extract_keywords(url)
    keywords = keywords_cleaner(keywords)
    keywords = [word for word in keywords if len(word) > 3]
    return " ".join(keywords)


def clear_product_text(text: str):
    words = text.split(" ")
    words = [word for word in words if not word.isnumeric() and len(word) > 3]
    return " ".join(words)


def process_text(url: str = None, product_text: str = None, picture_url: str = None):
    url_keywords_cleaned = pipeline_extract_data_4_url(url) if url else None
    picture_url_keywords_cleaned = None
    if picture_url:
        picture_url_keywords_cleaned = pipeline_extract_data_4_url(picture_url)
    product_text_cleaned = clear_product_text(product_text) if product_text else None
    total_text = ""

    if url_keywords_cleaned:
        total_text += url_keywords_cleaned + " "

    if picture_url_keywords_cleaned:
        total_text += " " + picture_url_keywords_cleaned

    if product_text_cleaned:
        total_text += " " + product_text_cleaned

    return total_text
