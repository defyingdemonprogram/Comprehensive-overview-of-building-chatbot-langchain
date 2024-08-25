import numpy as np

a = "This is a banana"
b = "Bananas are yellow"
c = "I like eating bananas"
docs = [a.split(), b.split(), c.split()]

N = len(docs)
avgdl = sum(len(sentence) for sentence in docs) / len(docs)

def bm25(word: str, sentence: str, docs: list,k: float=1.2, b: float=0.75):
  # term frequency -> f(f,D)
  freq = sentence.split().count(word)
  tf = (freq * (k+1)) / (freq + k*(1-b+b*(len(sentence.split()) / avgdl)))
  # inverse document frequency
  N_q = sum([1 for doc in docs if word in doc])  # number of docs that contain the word
  idf = np.log(((N-N_q + 0.5) / (N_q+0.5)) +1)
  return round(tf*idf, 4)

if __name__ == "__main__":
    word_to_search = "banana"
    print("TF-IDF score for '{}' in sentence 'a': {}".format(word_to_search, bm25(word_to_search, a, docs)))
