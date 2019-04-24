import pandas as pd

from entropy import calc_entropy

df = pd.read_csv(
    "tennis.txt",
    delimiter="\t",
    header=None,
    names=["outlook", "temperature", "humidity", "wind", "playing"],
)
print(df)

# Days where the wind is weak
print("\nGrabbing all days where the wind is weak")
print(df[df["wind"] == "Weak"])
print("\nEntropy of days where wind is weak")
print(calc_entropy(2 / 8))

# Days where the wind is strong
print("\nGrabbing all days where the wind is strong")
print(df[df["wind"] == "Strong"])
print("\nEntropy of days where wind is strong")
print(calc_entropy(3 / 6))

print("\nProbability that the wind is weak")
print(len(df[df["wind"] == "Weak"]) / len(df))

print("\nProbability that the wind is Strong")
print(len(df[df["wind"] == "Strong"]) / len(df))

winds = ["Weak", "Strong"]

decision_entropy = calc_entropy(5 / 14)
entropy = 0
for wind in winds:
    wind_prob = len(df[df["wind"] == wind]) / len(df)
    play_prob = len(df[df["wind"] == wind][df["playing"] == "Yes"]) / len(
        df[df["wind"] == wind]
    )
    wind_entropy = calc_entropy(play_prob)
    entropy += wind_entropy * wind_prob


def calculate_info_gain(df):
    cols = [col for col in df.columns if col != "playing"]
    
    for col in cols:


info_gain = decision_entropy - entropy
print(info_gain)
calculate_info_gain(df)
