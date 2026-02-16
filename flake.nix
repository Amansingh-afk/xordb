{
  description = "XDB - A blazingly fast Semantic Cache powered by Hyperdimensional Computing (HDC)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            go
            gopls
            gotools
            go-tools
            golangci-lint
          ];

          shellHook = ''
            echo ""
            echo "  ██╗  ██╗ ██████╗ ██████╗ ██████╗ ██████╗ "
            echo "  ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗"
            echo "   ╚███╔╝ ██║   ██║██████╔╝██║  ██║██████╔╝"
            echo "   ██╔██╗ ██║   ██║██╔══██╗██║  ██║██╔══██╗"
            echo "  ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝██████╔╝"
            echo "  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ "
            echo ""
            echo "  A blazingly fast Semantic Cache"
            echo "  powered by Hyperdimensional Computing"
            echo ""
            echo "  go run .           Run XDB"
            echo "  go build           Build binary"
            echo "  go test ./...      Run tests"
            echo ""
          '';

          CGO_ENABLED = "0";
        };
      }
    );
}
