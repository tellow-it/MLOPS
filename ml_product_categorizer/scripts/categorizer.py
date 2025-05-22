def find_closest_category(keywords: list[str], slug_categories: list[str]) -> None:
    best_distance = 0
    closest_category = None

    if not keywords:
        return None

    for category in slug_categories:
        iou = (
                len(set(keywords).intersection(set(category))) /
                len(set(keywords).union(set(category)))
        )
        if iou > best_distance:
            best_distance = iou
            closest_category = category


    return closest_category
