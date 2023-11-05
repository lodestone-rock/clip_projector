def flatten_dict(dictionary, parent_key='', separator='.'):
    """
    Flatten a nested dictionary into dot notation.
    
    Args:
    dictionary (dict): The nested dictionary to be flattened.
    parent_key (str): The concatenated keys from the parent dictionary.
    separator (str): The separator used in the dot notation.

    Returns:
    dict: The flattened dictionary in dot notation.
    """
    items = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, separator=separator))
        else:
            items[new_key] = value
    return items


def unflatten_dict(dictionary):
    """
    Unflatten a dictionary from dot notation back to its nested form.

    Args:
    dictionary (dict): The dictionary in dot notation.

    Returns:
    dict: The unflattened nested dictionary.
    """
    nested_dict = {}
    for key, value in dictionary.items():
        keys = key.split('.')
        temp_dict = nested_dict
        for k in keys[:-1]:
            temp_dict = temp_dict.setdefault(k, {})
        temp_dict[keys[-1]] = value
    return nested_dict
