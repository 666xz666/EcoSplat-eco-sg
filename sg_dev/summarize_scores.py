#!/usr/bin/env python3
"""Summarize scores_all.json files and pair them against a reference run.

Each scores_all.json maps scene hash to [overlap_label, psnr, ssim, lpips].
The first path on the command line is the reference; every later path is
paired against it on the shared keys, which is the same protocol the paper
tables use when two runs are compared scene by scene.

Args:
    paths: two or more scores_all.json paths; the first is the reference.

Returns:
    None. Prints one mean block per run and one paired block per comparison.
"""
import json
import statistics as st
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def main(argv):
    runs = [load(path) for path in argv]
    names = [path.split("/")[-2] for path in argv]
    for name, run in zip(names, runs):
        n = len(run)
        print("%-28s n=%d psnr=%.3f ssim=%.4f lpips=%.4f" % (
            name, n,
            st.mean(v[1] for v in run.values()),
            st.mean(v[2] for v in run.values()),
            st.mean(v[3] for v in run.values()),
        ))
    ref = runs[0]
    for name, run in zip(names[1:], runs[1:]):
        keys = sorted(set(ref) & set(run))
        if not keys:
            print("%s: no shared scenes" % name)
            continue
        for index, label, sign in ((1, "psnr", 1), (2, "ssim", 1), (3, "lpips", -1)):
            diff = [run[k][index] - ref[k][index] for k in keys]
            mean = st.mean(diff)
            se = st.pstdev(diff) / len(diff) ** 0.5
            better = sum(1 for d in diff if sign * d > 0)
            print("  %-28s %-5s mean=%+.4f se=%.4f t=%+.2f better=%d/%d" % (
                name, label, mean, se, mean / se, better, len(diff)))


if __name__ == "__main__":
    main(sys.argv[1:])
