def nearest_multiple(x, k):
    lower_multiple = (x // k) * k    
    upper_multiple = ((x + k - 1) // k) * k
    
    if abs(x - lower_multiple) <= abs(x - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple