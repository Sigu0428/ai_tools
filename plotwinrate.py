import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('_mpl-gallery')

winrates = []
for dir in os.listdir("deep_winrates/"):
    dirname = os.fsdecode(dir)
    winrate = np.loadtxt("deep_winrates/" + dirname + "/winrate.csv", delimiter=",")
    winrates.append(winrate)

Y = np.mean(np.array(winrates), axis=0)
X = np.linspace(0, 2000, len(Y))
Yp = np.ones(Y.shape)*0.552
Yp_low = np.ones(Y.shape)*0.520563
Yp_high = np.ones(Y.shape)*0.58313

y_err = np.std(np.array(winrates), axis=0)*2
print(np.array(winrates).shape[0])
#plt.errorbar(X, Y, yerr=y_err, ecolor='black', capsize=2)
fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))
ax.plot(X, Y)
ax.fill_between(X, Y-y_err, Y+y_err, alpha=0.3)
ax.plot(X, Yp, linestyle="dotted", color="red")
#ax.plot(X, Yp_high, linestyle="dotted", color="red")
ax.legend(["mean", "std. dev.", "priority agent"])
ax.set_xlabel("Ludo games")
ax.set_ylabel("win rate")
ax.set_title("Winrate during training")
ax.set_xlim(xmin=0, xmax=2000)
plt.savefig('winrate_deep.pdf', bbox_inches='tight')
plt.show()

winrates = []
for dir in os.listdir("tabular_winrates/"):
    dirname = os.fsdecode(dir)
    winrate = np.loadtxt("tabular_winrates/" + dirname + "/winrate.csv", delimiter=",")
    winrates.append(winrate)

Y = np.mean(np.array(winrates), axis=0)
X = np.linspace(0, 2000, len(Y))
Yp = np.ones(Y.shape)*0.552

y_err = np.std(np.array(winrates), axis=0)
print(np.array(winrates).shape[0])
#plt.errorbar(X, Y, yerr=y_err, ecolor='black', capsize=2)
fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))
ax.plot(X, Y)
ax.fill_between(X, Y-y_err, Y+y_err, alpha=0.3)
ax.plot(X, Yp, linestyle="dotted", color="red")
#ax.plot(X, Yp_high, linestyle="dotted", color="red")
ax.legend(["mean", "std. dev.", "priority agent"])
ax.set_xlabel("Ludo games")
ax.set_ylabel("win rate")
ax.set_title("Winrate during training")
ax.set_xlim(xmin=0, xmax=2000)
plt.savefig('winrate_tabular.pdf', bbox_inches='tight')
plt.show()