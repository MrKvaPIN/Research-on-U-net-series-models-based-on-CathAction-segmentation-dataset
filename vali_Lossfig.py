import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EPOCHS = 30

U_net = [
    0.2486, 0.2335, 0.2242, 0.2224, 0.2231, 0.2204, 0.2169, 0.2135,
    0.2121, 0.2161, 0.2146, 0.2083, 0.2114, 0.2092, 0.2086, 0.2138,
    0.2050, 0.2039, 0.2036, 0.2032, 0.2050, 0.2026, 0.2030, 0.2039,
    0.2012, 0.2021, 0.2021, 0.2009, 0.2011, 0.2007
]

R2U_net = [
    0.2899, 0.3229, 0.2766, 0.3127, 0.2573, 0.2535, 0.2865, 0.2698,
    0.2797, 0.2485, 0.2403, 0.2530, 0.2698, 0.2525, 0.2654, 0.2557,
    0.2434, 0.2635, 0.2533, 0.2513, 0.2586, 0.2546, 0.2540, 0.2595,
    0.2643, 0.2623, 0.2731, 0.2650, 0.2775, 0.2827
]

attention_unet = [
    0.2613, 0.2408, 0.2327, 0.2326, 0.2323, 0.2223, 0.2264, 0.2188,
    0.2160, 0.2126, 0.2180, 0.2188, 0.2105, 0.2095, 0.2141, 0.2069,
    0.2363, 0.2102, 0.2075, 0.2169, 0.2049, 0.2052, 0.2040, 0.2065,
    0.2067, 0.2043, 0.2042, 0.2035, 0.2036, 0.2032
]

attention_R2U_net = [
    0.2622, 0.2540, 0.2382, 0.2286, 0.2450, 0.2609, 0.2359, 0.2342,
    0.2357, 0.2626, 0.2749, 0.2743, 0.2718, 0.2916, 0.2881, 0.2820,
    0.3071, 0.3017, 0.3315, 0.3251, 0.3205, 0.3208, 0.3315, 0.3310,
    0.3426, 0.3383, 0.3425, 0.3486, 0.3466, 0.3388
]


epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, U_net, marker='o', markersize=3, linewidth=1.5, label='U-Net')
plt.plot(epochs, attention_unet, marker='s', markersize=3, linewidth=1.5, label='Attention U-Net')
plt.plot(epochs, attention_R2U_net, marker='^', markersize=3, linewidth=1.5, label='Attention R2U-Net')
plt.plot(epochs, R2U_net, marker='d', markersize=3, linewidth=1.5, label='R2U-Net')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss over Epochs', fontsize=14)

ax = plt.gca()
ax.set_xlim(3, 30)
yticks = np.arange(0, 0.43, 0.04)
ax.set_yticks(yticks)

plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_path = r'd:\FYP\ValidationLoss_comparison.png'
plt.savefig(output_path, dpi=150)
print(f'Saved to {output_path}')
plt.show()