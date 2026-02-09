#!/usr/bin/env python3
import sys
sys.path.insert(0, './src')

from ssr_ocl.neural.train_complete import run_complete_training

if __name__ == "__main__":
    run_complete_training()
