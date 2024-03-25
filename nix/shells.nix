{ inputs, ... }: {
  perSystem = { config, pkgs, ... }: {
    devShells = let
      greeting = "Deep Learning with Differentiable Logic";
      inherit (inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; })
        mkPoetryEnv;
      poetry-env = mkPoetryEnv {
        python = pkgs.python311;
        projectDir = "${inputs.self}/dl2-jax";
        preferWheels = true;
      };
    in {
      default = pkgs.mkShell {
        name = "dl2-replication-develop";
        shellHook = "echo ${greeting}";
        buildInputs = (import ./python.nix { inherit pkgs; })
          ++ (with pkgs; [ pkg-config libstdcxx5 zlib ]);
      };
      poetry = poetry-env.env.overrideAttrs
        (oldAttrs: { buildInputs = [ pkgs.python311Packages.ml-dtypes ]; });
    };
  };
}
