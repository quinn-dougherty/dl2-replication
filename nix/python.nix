{ pkgs }:
let
  py-deps = ps: with ps; [ ipython jupyter pytest black hypothesis ];
  dl2-deps = ps:
    with ps; [
      configargparse
      textx
      numpy
      scipy
      pillow
      torch
      torchvision
      matplotlib
      scikit-learn
    ];
  jax = ps:
    with ps; [
      jax
      jaxlib
      jaxlibWithoutCuda
      ml-dtypes
      jaxtyping
      einops
    ];
in [
  (pkgs.python311.withPackages
    (ps: builtins.concatLists (map (f: f ps) [ py-deps dl2-deps jax ])))
]
