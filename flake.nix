{
  description = "replication of Deep Learning with Differentiable Logic";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    parts.url = "github:hercules-ci/flake-parts";
    fmt = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { self, nixpkgs, parts, fmt, poetry2nix }@inputs:
    parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      imports =
        [ ./nix/shells.nix ./nix/dl2-jax.nix fmt.flakeModule ./nix/format.nix ];
      flake.herculesCI.ciSystems = [ "x86_64-linux" ];
    };
}
