import os
import importlib


def make_visualize(cfg):
    module=cfg.visualize_module
    visualize=importlib.import_module(module).Visualize()
    return visualize