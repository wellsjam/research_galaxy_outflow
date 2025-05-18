import matplotlib.pyplot as plt
import pyregion

reg_name = "ngc3741aeDeep1.reg"

ax = plt.subplot(111)
ax.set_xlim(171, 175)
ax.set_ylim(42.5, 46.5)
ax.set_aspect(1)

r = pyregion.open(reg_name)

patch_list, text_list = r.get_mpl_patches_texts()
for p in patch_list:
    ax.add_patch(p)
for t in text_list:
    ax.add_artist(t)

plt.draw()
plt.show()