import numpy as np

def levenshtein(a: str, b: str):
    # we must add an additional character at the start of each string
    a = f" {a}"
    b = f" {b}"
    # initialize the empty zero array
    lev = np.zeros((len(a), len(b)))
    # now iterate through each value, finding best path
    for i in  range(len(a)):
        for j in range(len(b)):
            if min([i, j]) == 0:
                lev[i, j] = max([i, j])
            else:
                # calculate three possivle outcome
                x = lev[i-1, j] # deletion
                y = lev[i, j-1] # insertion
                z = lev[i-1, j-1] # substitution
                # take the minimum (best operation/path)
                lev[i, j] = min([x, y, z])
                # and if our two current characters donot match, add 1
                if a[i] != b[j]:
                    lev[i, j] += 1
    # lLevenshtein disstance matrix and number of operation  need to be performed to make a=b(last element of matrix)
    return lev, lev[-1, -1]

if __name__=="__main__":    
    print(levenshtein("Levenshtein", "Levinsthen"))