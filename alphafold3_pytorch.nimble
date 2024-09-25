# Package

version       = "0.1.0"
author        = "Phil Wang"
description   = "Alphafold3"
license       = "MIT"
srcDir        = "alphafold3_pytorch"
bin           = @["alphafold3_pytorch"]


# Dependencies

requires "nim >= 2.0.8"
requires "nimpy >= 0.2.0"
requires "arraymancer"
requires "malebolgia"
