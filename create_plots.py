import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DPI = 200
aspect = (10, 8)

models = ['fastmri_knee_4_attention', 'fastmri_knee_4', 'fastmri_knee_3', 'fastmri_knee_2', 'fastmri_knee_1']
legend_entries = ['Depth = 4, Attention', 'Depth = 4', 'Depth = 3', 'Depth = 2', 'Depth = 1']
recos = ['CF_004', 'CF_004_AF_8', 'gaussian2d', 'gaussian2d_AF_8', 'radial', 'poisson_AF_15']
x_labels = ['Gaussian 1D 4x', 'Gaussian 1D 8x', 'Gaussian 2D 4x', 'Gaussian 2D 8x', 'Radial', 'Poisson 15x']
Ns = [250]

psnr_fastmri = np.zeros((len(models), len(recos)))
ssim_fastmri = np.zeros((len(models), len(recos)))

positions = np.array([-0.2, -0.1, 0, 0.1, 0.2])
colors = ['pink', 'lightblue', 'lightgreen', 'moccasin', 'lightgray']
data = []
data_positions = []
patches = []

# ------ CORPD PSNR -----------
plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

# mean_psnr_corpd = (np.array([np.mean(x) for x in data]).reshape((len(models), len(recos)))).T
# std_psnr_corpd = (np.array([np.std(x) for x in data]).reshape((len(models), len(recos)))).T


boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set normalized')
plt.tight_layout()
plt.savefig('psnr_corpd_norm.png', dpi=DPI)


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set normalized')
plt.tight_layout()
plt.savefig('psnr_corpdfs_norm.png', dpi=DPI)


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set normalized')
plt.tight_layout()
plt.savefig('ssim_corpd_norm.png', dpi=DPI)

# mean_ssim_corpd = (np.array([np.mean(x) for x in data]).reshape((len(models), len(recos)))).T
# std_ssim_corpd = (np.array([np.std(x) for x in data]).reshape((len(models), len(recos)))).T

# print(' &' + ' & '.join(legend_entries) + '\\\\\\hline')
# for i in range(len(recos)):
#     #for j in range(len(models)):
#     print(f'\multirow{"{2}"}{"{*}"}{"{" + x_labels[i] + "}"}')
#     print(' &' + ' & '.join([f'${mean:.2f} \pm {std:.2f}$' for mean, std in zip(mean_psnr_corpd[i, :], std_psnr_corpd[i, :])]) + '\\\\')
#     print(' &' + ' & '.join([f'${mean:.3f} \pm {std:.3f}$' for mean, std in zip(mean_ssim_corpd[i, :], std_ssim_corpd[i, :])]) + '\\\\\\hline')

# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set normalized')
plt.tight_layout()
plt.savefig('ssim_corpdfs_norm.png', dpi=DPI)


# ------ NMSE SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set normalized')
plt.tight_layout()
plt.savefig('nmse_corpd_norm.png', dpi=DPI)


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set normalized')
plt.tight_layout()
plt.savefig('nmse_corpdfs_norm.png', dpi=DPI)


# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig('psnr_corpd.png', dpi=DPI)
mean_psnr_corpd = np.array([np.mean(x) for x in data])

# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig('psnr_corpdfs.png', dpi=DPI)
mean_psnr_corpdfs = np.array([np.mean(x) for x in data])


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig('ssim_corpd.png', dpi=DPI)
mean_ssim_corpd = np.array([np.mean(x) for x in data])


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig('ssim_corpdfs.png', dpi=DPI)
mean_ssim_corpdfs = np.array([np.mean(x) for x in data])

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set')
plt.tight_layout()
plt.savefig('nmse_corpd.png', dpi=DPI)


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set')
plt.tight_layout()
plt.savefig('nmse_corpdfs.png', dpi=DPI)


# ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set normalized')
plt.tight_layout()
plt.savefig('psnr_brain_norm.png', dpi=DPI)


# ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set normalized')
plt.tight_layout()
plt.savefig('ssim_brain_norm.png', dpi=DPI)

# ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set normalized')
plt.tight_layout()
plt.savefig('nmse_brain_norm.png', dpi=DPI)


# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig('psnr_brain.png', dpi=DPI)
mean_psnr_brain = np.array([np.mean(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig('ssim_brain.png', dpi=DPI)
mean_ssim_brain = np.array([np.mean(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set')
plt.tight_layout()
plt.savefig('nmse_brain.png', dpi=DPI)



#--------------------- CelebA-HQ -------------------------------
models = ['celeba_4_attention', 'celeba_4', 'celeba_3', 'celeba_2', 'celeba_1']
# ------ CORPD PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_psnr_corpd, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_psnr_corpd.png', dpi=DPI)


# ------ CORPDFS PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_psnr_corpdfs, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_psnr_corpdfs.png', dpi=DPI)


# ------ CORPD SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_ssim_corpd, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_ssim_corpd.png', dpi=DPI)


# ------ CORPDFS SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_ssim_corpdfs, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_ssim_corpdfs.png', dpi=DPI)

# ------ CORPD NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPD data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_nmse_corpd.png', dpi=DPI)


# ------ CORPDFS NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_FS' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('CORPDFS data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_nmse_corpdfs.png', dpi=DPI)

# # ------ Brain PSNR -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/psnr_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_psnr_brain, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('PSNR in dB')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_psnr_brain.png', dpi=DPI)
mean_psnr_brain = np.array([np.mean(x) for x in data])


# # ------ Brain SSIM -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/ssim_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
h = plt.scatter(data_positions, mean_ssim_brain, c='red', marker='x', zorder=3, label='In-distribution means')
patches.append(h)
plt.legend(handles=patches)
plt.ylabel('SSIM in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_ssim_brain.png', dpi=DPI)
mean_ssim_brain = np.array([np.mean(x) for x in data])

# # ------ Brain NMSE -----------
data = []
data_positions = []
patches = []

plt.figure(figsize=aspect)
for j, model in enumerate(models):
    for i, reco in enumerate(recos):
        data.append(np.loadtxt('results/' + model + '_' + reco + '_N_' + str(Ns[0]) + '_brain' + '_no_norm/nmse_values.csv', delimiter=','))
        data_positions.append(i + 1 + 1.5 * positions[j])
    patches.append(Patch(facecolor=colors[j], label=legend_entries[j]))

boxes = plt.boxplot(data, widths = 0.1, positions=data_positions, patch_artist=True, flierprops=dict(marker='+'))
for i, box in enumerate(boxes['boxes']):
    box.set(facecolor=colors[i // len(recos)])
plt.gca().set_xticks(range(1, len(recos)+1))
plt.gca().set_xticklabels(x_labels)
plt.legend(handles=patches)
plt.ylabel('NMSE in a.u.')
plt.title('Brain data set, out-of-distribution training')
plt.tight_layout()
plt.savefig('celeba_nmse_brain.png', dpi=DPI)