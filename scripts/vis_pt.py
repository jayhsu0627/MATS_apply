import torch
import matplotlib.pyplot as plt

# Load your vector
vector = torch.load("/mnt/drive_b/MATS_apply/analogy_vector.pt") # Shape [42, 3584]
# norms = vector.norm(dim=1).cpu().numpy()
norms = vector.norm(dim=1).to(dtype=torch.float32).cpu().numpy()

# Print norms for middle layers to identify a better candidate
print("\nVector Norms per layer:")
for layer, norm in enumerate(norms):
    print(f"Layer {layer}: {norm:.4f}")

# Plotting (Optional, if you have X11 forwarding or just want to save the image)
plt.plot(norms)
plt.title("Analogy Vector Divergence by Layer")
plt.xlabel("Layer")
plt.ylabel("Euclidean Norm")
plt.savefig("/mnt/drive_b/MATS_apply/layer_norms.png")
print("\nCheck layer_norms.png. Look for a 'bump' in layers 20-35.")