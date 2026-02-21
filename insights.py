def compare_segments(segment_results, metric="expectancy", threshold=0.15):
    if len(segment_results) < 2:
        return None

    sorted_items = sorted(
        segment_results.items(),
        key=lambda x: x[1][metric],
        reverse=True,
    )

    best_name, best_data = sorted_items[0]
    worst_name, worst_data = sorted_items[-1]

    best_value = best_data[metric]
    worst_value = worst_data[metric]

    if best_value == 0:
        return None

    diff_ratio = abs(best_value - worst_value) / abs(best_value)

    if diff_ratio < threshold:
        return None

    return best_name, worst_name


def generate_insight(segment_results, category_name):
    comparison = compare_segments(segment_results)

    if comparison is None:
        return None

    best, worst = comparison

    return (
        f"You perform significantly better in {best} "
        f"{category_name} compared to {worst}."
    )