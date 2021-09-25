def wavg(group, avg_name, weight_name):
    """
    Computes weighted averages by group.

    Args:
        group (Pandas DataFrame): data for the particular grouping
        avg_name (string): name of variable to compute wgt avg with
        weight_name (string): name of weighting variables

    Returns:
        d (scalar): weighted avg for the group
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except Warning:
        return d.mean()
