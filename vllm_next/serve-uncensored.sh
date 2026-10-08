#!/usr/bin/env bash
# serve.sh with the CRACK-edited (uncensored) GLM-5.3 AWQ checkpoint.
# Everything else (profiles, drafter, env) is identical to serve.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=${MODEL:-$HOME/models/GLM-5.3-UNCENSORED-Int4-Int8Mix-AWQ-g64}
exec "$HERE/serve.sh" "$@"
