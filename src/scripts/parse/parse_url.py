import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def extract_product_info(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Failed to load URL: {e}")
        return None

    # Title extraction
    title = (
            soup.find("meta", property="og:title") or
            soup.find("meta", attrs={"name": "title"}) or
            soup.title
    )
    title = title.get("content") if title and title.has_attr("content") else getattr(title, "text", "").strip()

    # Description extraction
    description = (
            soup.find("meta", property="og:description") or
            soup.find("meta", attrs={"name": "description"})
    )
    description = description.get("content") if description else ""

    # Image extraction (try og:image first)
    image = soup.find("meta", property="og:image")
    image_url = image.get("content") if image else ""

    if not image_url:
        # Fallback to largest visible image
        images = soup.find_all("img")
        image_candidates = []
        for img in images:
            src = img.get("src")
            if not src or "logo" in src or "icon" in src:
                continue
            full_url = urljoin(url, src)
            width = int(img.get("width", 0)) if img.get("width", "").isdigit() else 0
            height = int(img.get("height", 0)) if img.get("height", "").isdigit() else 0
            image_candidates.append((width * height, full_url))

        if image_candidates:
            image_candidates.sort(reverse=True)  # biggest first
            image_url = image_candidates[0][1]

    return {
        "url": url,
        "title": title.strip(),
        "description": description.strip(),
        "image_url": image_url.strip()
    }
