import os
import importlib


def make_renderer(cfg, net):
    module = cfg.renderer_module
    renderer = importlib.import_module(module).Renderer(net)
    return renderer
