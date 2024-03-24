{ ... }: {
  perSystem = { ... }: {
    treefmt.config = {
      projectRootFile = "flake.nix";
      programs = {
        nixfmt.enable = true;
        prettier.enable = true;
        black.enable = true;
      };
      settings.formatter = {
        prettier.excludes = [ "original-source/" ];
        black.excludes = [ "original-source/" ];
      };
    };
  };
}
