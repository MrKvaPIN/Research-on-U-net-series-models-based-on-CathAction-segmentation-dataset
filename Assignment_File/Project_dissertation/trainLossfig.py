import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EPOCHS = 30

U_net = [
    0.2827, 0.2408, 0.2325, 0.2283, 0.2245, 0.2202, 0.2185, 0.2168,
    0.2149, 0.2137, 0.2117, 0.2106, 0.2092, 0.2096, 0.2077, 0.2079,
    0.2042, 0.2034, 0.2027, 0.2024, 0.2021, 0.2023, 0.2016, 0.2009,
    0.2010, 0.2003, 0.1998, 0.2001, 0.1994, 0.1996
]

R2U_net = [
    0.2999, 0.2477, 0.2360, 0.2304, 0.2276, 0.2248, 0.2218, 0.2199,
    0.2163, 0.2162, 0.2131, 0.2120, 0.2118, 0.2106, 0.2099, 0.2054,
    0.2046, 0.2043, 0.2038, 0.2011, 0.2009, 0.2009, 0.2001, 0.1993,
    0.1988, 0.1984, 0.1988, 0.1977, 0.1972, 0.1972
]

attention_unet = [
    0.3657, 0.2601, 0.2470, 0.2410, 0.2380, 0.2350, 0.2322, 0.2303,
    0.2288, 0.2269, 0.2260, 0.2248, 0.2231, 0.2211, 0.2214, 0.2195,
    0.2200, 0.2191, 0.2190, 0.2179, 0.2144, 0.2142, 0.2134, 0.2127,
    0.2132, 0.2121, 0.2115, 0.2107, 0.2104, 0.2098
]

attention_R2U_net = [
    0.3125, 0.2564, 0.2475, 0.2428, 0.2391, 0.2376, 0.2354, 0.2329,
    0.2314, 0.2287, 0.2277, 0.2270, 0.2197, 0.2188, 0.2184, 0.2182,
    0.2152, 0.2144, 0.2139, 0.2110, 0.2111, 0.2110, 0.2105, 0.2093,
    0.2090, 0.2089, 0.2089, 0.2094, 0.2084, 0.2081
]


epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, U_net, marker='o', markersize=3, linewidth=1.5, label='U-Net')
plt.plot(epochs, attention_unet, marker='s', markersize=3, linewidth=1.5, label='Attention U-Net')
plt.plot(epochs, attention_R2U_net, marker='^', markersize=3, linewidth=1.5, label='Attention R2U-Net')
plt.plot(epochs, R2U_net, marker='d', markersize=3, linewidth=1.5, label='R2U-Net')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss over Epochs', fontsize=14)

ax = plt.gca()
ax.set_xlim(3, 30)
yticks = np.arange(0, 0.43, 0.04)
ax.set_yticks(yticks)

plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_path = r'd:\FYP\TrainingLoss_comparison.png'
plt.savefig(output_path, dpi=150)
print(f'Saved to {output_path}')
plt.show()