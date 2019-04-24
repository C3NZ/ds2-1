import numpy as np


# Given the first Probability of our coin, calculate the entropy
def calc_entropy(P):
    return -(P * np.log2(P) + (1 - P) * np.log2(1 - P))


# Calculate the entropy of our 3 coins
print(calc_entropy(0.5))
print(calc_entropy(0.9))
print(calc_entropy(0.1))

# Calculate the entropy of someone playing tennis from our tennis dataset
# (9 out of 14 people said no)
print(calc_entropy(9 / 14))
