from torch.nn import Module

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class Alphafold3(Module):
    def __init__(self):
        super().__init__()
