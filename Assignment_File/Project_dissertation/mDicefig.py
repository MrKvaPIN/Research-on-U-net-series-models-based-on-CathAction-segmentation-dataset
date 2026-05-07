import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EPOCHS = 30

U_net = [
    0.6092, 0.6322, 0.6472, 0.6493, 0.6488, 0.6520, 0.6579, 0.6634,
    0.6650, 0.6582, 0.6621, 0.6706, 0.6654, 0.6689, 0.6698, 0.6614,
    0.6752, 0.6770, 0.6777, 0.6782, 0.6751, 0.6792, 0.6788, 0.6772,
    0.6812, 0.6798, 0.6799, 0.6818, 0.6818, 0.6821
]

R2U_net = [
    0.5653, 0.5186, 0.5634, 0.5495, 0.6266, 0.6131, 0.6265, 0.6266,
    0.5669, 0.6374, 0.6394, 0.6334, 0.6116, 0.6294, 0.6058, 0.6076,
    0.6253, 0.5882, 0.6158, 0.6324, 0.6135, 0.6192, 0.6223, 0.6249,
    0.6086, 0.6137, 0.5828, 0.6035, 0.5883, 0.5767
]

attention_unet = [
    0.5980, 0.6207, 0.6319, 0.6477, 0.6359, 0.6413,
    0.6534, 0.6575, 0.6635, 0.6546, 0.6532, 0.6577, 0.6665, 0.6656,
    0.6680, 0.6603, 0.6631, 0.6721, 0.6766, 0.6726, 0.6752, 0.6723,
    0.6734, 0.6762, 0.6765, 0.6751, 0.6752, 0.6776, 0.6772, 0.6777
]

attention_R2U_net = [
    0.5977, 0.6158, 0.6292, 0.6423, 0.6224, 0.6064, 0.6431, 0.6504,
    0.6511, 0.6302, 0.6486, 0.6466, 0.6516, 0.6415, 0.6314, 0.6494,
    0.6391, 0.6371, 0.6444, 0.6367, 0.6361, 0.6513, 0.6417, 0.6419,
    0.6476, 0.6487, 0.6408, 0.6394, 0.6453, 0.6303
]


epochs = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, U_net, marker='o', markersize=3, linewidth=1.5, label='U-Net')
plt.plot(epochs, attention_unet, marker='s', markersize=3, linewidth=1.5, label='Attention U-Net')
plt.plot(epochs, attention_R2U_net, marker='^', markersize=3, linewidth=1.5, label='Attention R2U-Net')
plt.plot(epochs, R2U_net, marker='d', markersize=3, linewidth=1.5, label='R2U-Net')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('mDice', fontsize=12)
plt.title('Validation mDice over Epochs', fontsize=14)

ax = plt.gca()
ax.set_xlim(3, 30)
ax.set_ylim(0.46, 0.72)
yticks = np.arange(0.50, 0.73, 0.04)
ax.set_yticks(yticks)

plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

output_path = r'd:\FYP\mDice_comparison.png'
plt.savefig(output_path, dpi=150)
print(f'Saved to {output_path}')
plt.show()
