import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

# === 1. 读取 npz 掩码 ===
mask = np.load(r"data/leaf_mask_256_scaled.npz")["mask"].astype(np.float32)
# mask = data[list(data.keys())[0]].astype(np.float32)

print("mask shape:", mask.shape, "value range:", mask.min(), mask.max())

# === 2. 建立 taichi 场 ===
boundary_mask = ti.field(dtype=ti.f32, shape=mask.shape)
boundary_mask.from_numpy(mask)

# === 3. 可视化 ===
plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap="gray", origin="lower")  # 直接可视化
plt.title("Loaded Mask from leaf_mask_256.npz")
plt.axis("off")
plt.show()
