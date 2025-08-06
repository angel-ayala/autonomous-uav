#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:32:16 2025

@author: timevisao
"""
import subprocess
import sys


def run_script(exp_path, output_file='computational_cost.txt', error_file='error.txt'):
    # Full command: Python interpreter + script + arguments
    cmd = [sys.executable, "cost_profile.py",
           "--logspath", exp_path, "--with-model"]

    # Run and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate()

    if stderr and not 'Could not deserialize' in stderr:
        print(f"Error with {exp_path}")

        # Save to file
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(str(exp_path) + "\n")
            f.write(stderr)
            f.write("\n" + "#" * 20 + "\n")
    else:
        # Save to file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(stdout)
            f.write("#" * 20 + "\n")

        print(f"{exp_path}'s computational cost saved to {output_file}")


if __name__ == "__main__":
    from pathlib import Path
    from natsort import humansorted

    base_path = Path('logs_cf_proprio')
    output_file = 'computational_cost_proprio.txt'
    error_file = 'computational_cost_proprio_error.txt'

    exp_paths = []
    algs = []
    for d in base_path.iterdir():
        if 'assets' in d.name or 'alm' in d.name or d.is_file():
            continue
        exp_paths.append(d)
    exp_paths = humansorted(exp_paths)

    for d in exp_paths:
        alg_name = d.name.split('_')[0]
        if alg_name not in algs:
            run_script(d.relative_to(base_path.parent), output_file, error_file)
            algs.append(alg_name)
