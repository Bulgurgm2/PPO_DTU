import matplotlib.pyplot as plt
import pandas as pd
import os


top = os.getcwd()
f = []
for (dirpath, dirnames, filenames) in os.walk(top):
    print(dirpath)
    f.extend(filenames)
    # f.extend(os.path.join(top, dirpath, filenames))
    break
name = ""
for s in f:
    if "sample_" not in s:
        continue
    csv = pd.read_csv(s)
    print(csv)
    name = s.replace("sample_", "").replace("_", " ").replace(".csv", "")
    plt.semilogx(csv['step'],
                 [float(str(x).replace("tensor(", "").replace(")", "")) for x in csv['train']],
                 label=name)
#
plt.xlabel("Steps")
plt.ylabel("Reward")

plt.legend(loc='upper left')
#plt.grid()

for i in range(2000):
    name = f"{name.replace(' ', '_').split('=')[0]}.png"
    if os.path.isfile(os.path.join(os.getcwd(), name)):
        continue
    else:
        plt.savefig(name)
        print("Graph ", i, " saved")
        plt.clf()
        break



