#! /usr/bin/env python

from randmst import *
from matplotlib.ticker import AutoLocator, ScalarFormatter
import pandas as pd

results_dict = {}

ns_to_test = [2**k for k in range(2,14)]
dims_to_test = [0,2,3,4]

results_dict["num vertices"] = ns_to_test

fix, ax = plt.subplots()

for dim in dims_to_test:
    results = [run_experiment(num_trials=10, num_points=n, dim=dim, prune=True)
            for n in ns_to_test]
    ax.plot(ns_to_test, results, marker='o')

    col_name = "uniform" if dim == 0 else f"dimension {dim}"
    results_dict[col_name] = results

df = pd.DataFrame(results_dict)
print(df)

df.to_csv("results.csv")

plt.title(f"Average weight of MST for random unit cube graph")

plt.xlabel(f"number of vertices")
plt.ylabel(f"average weight")
plt.xscale("log", base=2)
plt.yscale("log", base=2)

filename = "avg weights.png"

plt.legend(["uniform"] + \
        [f"dimension {dim}" for dim in dims_to_test[1:]]
            )

plt.savefig(filename)
plt.show()
