

def annualized_price(price_ngn, price_period):
    if price_ngn is None:
        return None
    try:
        price = float(price_ngn)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    if price_period and str(price_period).lower() == "month":
        return price * 12
    return price


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = pct / 100 * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def median(values):
    vals = sorted([float(v) for v in values if v is not None])
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def range_stats(values, percentile_min_count=10):
    vals = sorted([float(v) for v in values if v is not None])
    if not vals:
        return None, None, "min_max"
    if len(vals) >= percentile_min_count:
        return _percentile(vals, 10), _percentile(vals, 90), "percentile"
    return vals[0], vals[-1], "min_max"


def classify(count, low_data_max=3, limited_max=9):
    if count == 0:
        return "no_data", "insufficient"
    if count <= low_data_max:
        return "low_data", "insufficient"
    if count <= limited_max:
        return "success", "limited"
    return "success", "normal"


def bucket_for(bedrooms, property_type):
    pt = (property_type or "").lower()
    if "self" in pt:
        return "self_contain"
    if bedrooms == 1:
        return "one_bedroom"
    if bedrooms == 2:
        return "two_bedroom"
    if bedrooms == 3:
        return "three_bedroom"
    return None


def summarize_values(values, percentile_min_count=10):
    vals = sorted([float(v) for v in values if v is not None and float(v) > 0])
    if not vals:
        return None
    mn, mx, method = range_stats(vals, percentile_min_count)
    return {
        "min": round(mn, 2) if mn is not None else None,
        "max": round(mx, 2) if mx is not None else None,
        "median": round(median(vals), 2),
        "count": len(vals),
        "range_method": method,
    }
