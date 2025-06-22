import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib as mpl
import re

#for i in range(10):
#    winrate = np.loadtxt("gamma_exp_results/winrate (" + str(i) + ").csv", delimiter=",")
#    winr = np.cumsum(np.array(winrate), dtype=float)
#    winr[n:] = (winr[n:] - winr[:-n])
#    winr = winr[n-1:] / n
#    plt.plot(winr)
#plt.legend(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
#plt.show()

from matplotlib import cm

plt.style.use('_mpl-gallery')

winrates = {}
for dir in os.listdir("gamma_exp_results_runs/"):
    dirname = os.fsdecode(dir)
    gamma = re.search("[\d.]+$", dirname).group(0)
    winrate = np.loadtxt("gamma_exp_results_runs/" + dirname + "/winrate.csv", delimiter=",")
    if gamma in winrates.keys():
        winrates[gamma].append(winrate)
    else:
        winrates[gamma] = [winrate]

# Make data
cmap = matplotlib.cm.get_cmap('brg')
fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))
for gamma in winrates.keys():
    #print(len(winrates[gamma]))
    Y = np.mean(np.array(winrates[gamma])[0:16,:], axis=0)
    X = np.linspace(0, 1, Y.shape[0])*2000
    rgba = cmap(float(gamma))
    ax.plot(X, Y, c=rgba)
#ax.set_xlabel("episodes")
#ax.set_ylabel("gamma")
#ax.set_zlabel("winrate")
ax.set_xlabel("episodes")
ax.set_ylabel("winrate")
ax.set_title("win rate over Ludo games for different discount values")
#plt.legend(["0.0", "0.1", "0.2", "0.3", "0.4", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='brg'),
             ax=ax, orientation='vertical', label='discount')
plt.savefig('gamma_experiment.pdf', bbox_inches='tight')
plt.show()