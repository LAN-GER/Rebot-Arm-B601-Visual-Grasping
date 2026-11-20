# SDK checkout and model download helpers used by setup_env.sh.
# Only publish a download after checking its content. A retry keeps the destination usable.
install_model() (
    local dest=$1 url=$2 kind=${3:-model} source_file=${4:-} part
    if works "$kind" "$dest"; then return; fi
    mkdir -p "$(dirname "$dest")"
    part=$(mktemp "${dest}.part.XXXXXX")
    trap 'rm -f -- "$part"' EXIT
    if [[ -n "$source_file" ]]; then
        cp -- "$source_file" "$part"
    else
        curl -fLsS --retry 3 --connect-timeout 20 --speed-time 45 --speed-limit 1024 -o "$part" -- "$url"
    fi
    probe "$kind" "$part"
    mv -- "$part" "$dest"
)

clone_sdk() {
    local dest=$1 marker=$2 url=$3
    shift 3
    if [[ -f "$dest/$marker" ]]; then return; fi
    if [[ -e "$dest" && ! -d "$dest" ]] || [[ -d "$dest" && -n "$(ls -A -- "$dest")" ]]; then
        echo "Incomplete SDK at $dest (missing $marker). Restore the SDK there or select a new path in the config." >&2
        return 1
    fi
    mkdir -p "$(dirname "$dest")"
    git clone "$@" "$url" "$dest"
    if [[ ! -f "$dest/$marker" ]]; then
        echo "The downloaded SDK is missing $dest/$marker. Check the upstream version." >&2
        return 1
    fi
}
