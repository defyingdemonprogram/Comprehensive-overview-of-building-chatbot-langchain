def get_shingles(text, w):
    """
    Generate shingles (substrings) of length `w` from the given text.
    
    :param text: Input string to generate shingles from.
    :param w: Length of each shingle.
    :return: Set of shingles.
    """
    shingles = set()
    text = text.lower()  # Convert text to lowercase for case-insensitive comparison
    for i in range(len(text) - w + 1):
        shingle = text[i:i + w]
        shingles.add(shingle)
    return shingles

def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.
    
    :param set1: First set of shingles.
    :param set2: Second set of shingles.
    :return: Jaccard similarity score.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def text_similarity(text1, text2, w):
    """
    Calculate the similarity between two texts using w-shingling and Jaccard similarity.
    
    :param text1: First input text.
    :param text2: Second input text.
    :param w: Length of each shingle.
    :return: Similarity score between 0 and 1.
    """
    shingles1 = get_shingles(text1, w)
    shingles2 = get_shingles(text2, w)
    return jaccard_similarity(shingles1, shingles2)

# Example usage
if __name__ == "__main__":
    text1 = "This is a sample text for testing similarity."
    text2 = "This sample text is used to test similarity."
    print(f"Text1: {text1}")
    print(f"Text2: {text2}")

    w = 5  # Shingle length

    similarity_score = text_similarity(text1, text2, w)
    print(f"Text similarity score: {similarity_score:.2f}")
