import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="darkgrid")

scaler = StandardScaler()

df = pd.read_csv("out/crypto/daily_20_2189_marked.csv")[["Date", "ADA-USD", "BTC-USD", "ETH-USD",
                                                         "XTZ-USD"]].set_index("Date")[-730:]
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

df.plot(ax=ax)
plt.ylabel("Standardized Price")

middle_line = len(df) // 2
plt.axvline(x=middle_line, color="gray", linestyle="--")  # Changed color to dark gray

bull_text_x = middle_line / 2
bear_text_x = middle_line + (len(df) - middle_line) / 2

plt.text(
    bull_text_x,
    ax.get_ylim()[1] * 0.75,
    "Bull",
    horizontalalignment="center",
    color="green",
    fontsize=18,
    fontweight="bold",
)
plt.text(
    bear_text_x,
    ax.get_ylim()[1] * -0.4,
    "Bear",
    horizontalalignment="center",
    color="red",
    fontsize=18,
    fontweight="bold",
)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("cryptoBullBear.svg")
