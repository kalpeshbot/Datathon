# -*- coding: utf-8 -*-
"""
LAPE Currency and Number Formatting Module
Provides shared, reproducible Indian Rupee (INR - ₹) formatting with Indian number grouping.
Examples: ₹1,250.00, ₹12,34,567.89, ₹5,01,04,056.90
"""
import math
import pandas as pd


def format_inr(amount, precision: int = 2) -> str:
    """
    Format a numeric amount as Indian Rupees (₹) with Indian digit grouping:
    - Last 3 digits grouped together
    - Preceding digits grouped in sets of 2
    Examples:
        1250.0 -> '₹1,250.00'
        1234567.89 -> '₹12,34,567.89'
        50104056.9 -> '₹5,01,04,056.90'
        -4500.5 -> '-₹4,500.50'
    """
    if amount is None or pd.isna(amount):
        return "-"
    
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return str(amount)

    if math.isnan(val) or math.isinf(val):
        return "-"

    is_negative = val < 0
    abs_val = abs(val)

    if precision > 0:
        formatted_float = f"{abs_val:.{precision}f}"
        int_str, frac_str = formatted_float.split(".")
    else:
        int_str = str(int(round(abs_val)))
        frac_str = ""

    # Indian digit grouping on integer portion
    if len(int_str) <= 3:
        grouped_int = int_str
    else:
        last3 = int_str[-3:]
        remaining = int_str[:-3]
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
        groups.reverse()
        grouped_int = ",".join(groups) + "," + last3

    res = f"₹{grouped_int}"
    if frac_str:
        res += f".{frac_str}"

    return f"-{res}" if is_negative else res


def format_inr_compact(amount) -> str:
    """
    Format a numeric amount into compact Indian financial denominations:
    - Crores (Cr) for >= 1,00,00,000
    - Lakhs (L) for >= 1,00,000
    - Thousands (k) for >= 1,000
    """
    if amount is None or pd.isna(amount):
        return "-"
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return str(amount)

    abs_val = abs(val)
    prefix = "-" if val < 0 else ""

    if abs_val >= 1e7:
        return f"{prefix}₹{abs_val / 1e7:.2f} Cr"
    elif abs_val >= 1e5:
        return f"{prefix}₹{abs_val / 1e5:.2f} L"
    elif abs_val >= 1e3:
        return f"{prefix}₹{abs_val / 1e3:.1f} k"
    else:
        return format_inr(val, precision=2)
