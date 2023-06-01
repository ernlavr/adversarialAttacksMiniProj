import pickle
import numpy as np
import string
from dataclasses import dataclass

@dataclass
class DP:
    toLabel : int
    count : int
    



# load in the samples.pkl file
with open('samples.pkl', 'rb') as f:
    samples = pickle.load(f)

# Find and count the number of each word that has been substititued between the original and adversarial text
substitutions = {}
topSubs = 10
fails = 0
for i, sample in enumerate(samples):
    fromLabel = sample[0]
    toLabel = sample[1]
    origStr = np.array(sample[2].translate(str.maketrans('', '', string.punctuation)).lower().split())
    advStr = np.array(sample[3].translate(str.maketrans('', '', string.punctuation)).lower().split())
    if (len(origStr) != len(advStr)):
        fails += 1
        print(i)
        continue
    # Find the words that have been substituted
    diff = np.where(origStr != advStr)[0]
    # fetch the diff indices from advStr and put them insubstitutions against toLabel
    for d in diff:
        token = advStr[d]
        if token in substitutions:
            if toLabel in [x.toLabel for x in substitutions[token]]:
                for s in substitutions[token]:
                    if s.toLabel == toLabel:
                        s.count += 1
            else:
                substitutions[token].append(DP(toLabel, 1))
        else:
            substitutions[token] = [DP(toLabel, 1)]

# for each substitution, get all dp that  have their toLabel equal to 0
# and add them to the count of the dp that has toLabel equal to 1
def sortSubs(targetLabel):
    zeros = []
    for s in substitutions.items():
        for dp in s[1]:
            if dp.toLabel == targetLabel:
                zeros.append((s[0], dp.count))
    # order zeros by the third element
    zeros = sorted(zeros, key=lambda x: x[1], reverse=True)
    return zeros

for label in range(3):
    pm = sortSubs(label)[:topSubs]
    print("Label: ", label)
    for l in pm:
        print(l)
    print("")

print(fails)





