import numpy as np

a = "This is a banana"
b = "Bananas are yellow"
c = "I like eating bananas"
docs = [a.split(), b.split(), c.split()]

def tf_idf(word: str, sentence: str, docs: list):
    # Term frequency (search in a document)
    # tf = sentence.count(word) / len(sentence.split())
    # we want to match exact word only
    tf = sentence.split().count(word) / len(sentence.split())
    
    # Calculate IDF only if the word is present in at least one document
    if any(word in doc for doc in docs):
        # Inverse document frequency (search in all documents)
        idf = np.log10(len(docs) / sum([1 for doc in docs if word in doc]))
        return round(tf * idf, 4)
    else:
        # If the word is not present in any document, return IDF as 0
        return 0.0


if __name__ == "__main__":
    word_to_search = "banana"
    print("TF-IDF score for '{}' in sentence 'a': {}".format(word_to_search, tf_idf(word_to_search, a, docs)))
