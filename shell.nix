{ pkgs ? import <nixpkgs> {}}:
let

  fhs = pkgs.buildFHSUserEnv {
    name = "llava-med-fhs-environment";

    targetPkgs = _: (with pkgs; [
      python311
      python311Packages.pip
      clang
      rustup
      pkg-config
    ]);

    buildInputs = with pkgs; [
      openssl
    ];

    profile = ''
      set -e
      rustup toolchain install stable
      rustup default stable
      python -m venv ./venv
      source venv/bin/activate
      export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig";
      pip install --upgrade pip  # enable PEP 660 support
      pip install -e .
      if [ -f .env ]; then
        source .env
      fi
      set +e
    '';
  };
in fhs.env

