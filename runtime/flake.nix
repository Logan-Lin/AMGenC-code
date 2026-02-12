{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
  in {
    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          uv python312 mongosh rsync zlib
          stdenv.cc.cc.lib  # provides libstdc++.so.6 for Python packages with C extensions
          (writeShellScriptBin "sync-remote" ''
            set -e
            if [ -z "$REMOTE_HOSTS" ]; then
              echo "Error: REMOTE_HOSTS not set. Add it to .env (e.g., REMOTE_HOSTS=\"user@server1:/path user@server2:/path\")"
              exit 1
            fi

            pids=()
            for host in $REMOTE_HOSTS; do
              echo "Syncing to $host..."
              rsync -avhP \
                --perms \
                --timeout=120 \
                -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3" \
                --include '.env' \
                --filter=':- .gitignore' \
                --exclude '.git' \
                . "$host" &
              pids+=($!)
            done

            # Wait for all syncs to complete
            failed=0
            for pid in "''${pids[@]}"; do
              if ! wait "$pid"; then
                failed=1
              fi
            done

            if [ "$failed" -eq 1 ]; then
              echo "One or more syncs failed"
              exit 1
            fi
            echo "All syncs completed successfully"
          '')
          (writeShellScriptBin "add-run" ''
            streamlit run helper/add_run.py
          '')
          (writeShellScriptBin "find-run" ''
            streamlit run helper/find_run.py
          '')
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          export UV_PYTHON=${pkgs.python312}/bin/python3
          uv sync
          source .venv/bin/activate

          # Load all vars from .env if it exists
          if [ -f .env ]; then
            while IFS='=' read -r key value; do
              # Strip surrounding quotes from value
              value="''${value#\"}"
              value="''${value%\"}"
              value="''${value#\'}"
              value="''${value%\'}"
              [ -n "$key" ] && export "$key=$value"
            done < <(grep -v '^#' .env | grep -v '^$')
          fi
        '';
      };
    });
  };
}
