def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    
    Parameters:
    set1 (set): The first set.
    set2 (set): The second set.
    
    Returns:
    float: The Jaccard similarity coefficient.
    """
    # Calculate the intersection of the two sets
    intersection = set1.intersection(set2)
    
    # Calculate the union of the two sets
    union = set1.union(set2)
    
    # Compute Jaccard similarity
    if not union:  # handle the case where both sets are empty
        return 1.0
    return len(intersection) / len(union)

# Example usage
if __name__ == "__main__":
    # Define two example sets
    set_a = {"apple", "banana", "cherry"}
    set_b = {"banana", "cherry", "date"}

    # Compute Jaccard similarity
    similarity = jaccard_similarity(set_a, set_b)
    
    print(f"Jaccard Similarity between {set_a} and {set_b} is {similarity:.2f}")

