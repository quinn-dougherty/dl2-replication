{ inputs, ... }: {
  perSystem = { config, pkgs, ... }: {
    devShells = let greeting = "Deep Learning with Differentiable Logic";
                    inherit (inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryEnv;
    in {
      default = pkgs.mkShell {
        name = "dl2-replication-develop";
        shellHook = "echo ${greeting}";
        buildInputs = (import ./python.nix { inherit pkgs; }) ++ (with pkgs; [ pkg-config libstdcxx5 zlib ]);
      };
      poetry = mkPoetryEnv {
        python = pkgs.python311;
        projectDir = "${inputs.self}/dl2-jax";
      };
    };
  };
}
