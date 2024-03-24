{ inputs, ... }: {
  perSystem = { config, pkgs, ... }:
    let
      inherit (inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; })
        mkPoetryEditablePackage;
      dl2-jax = mkPoetryEditablePackage {
        projectDir = "${inputs.self}/dl2-jax";
        editablePackageSources.dl2-jax = "${inputs.self}/dl2-jax";
      };
    in { packages = { inherit dl2-jax; }; };
}
