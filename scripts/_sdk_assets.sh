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
    if [[ -n ${REBOT_INSTALL_JOURNAL:-} ]]; then
        printf '[model] Installed %s\n' "$dest" >> "$REBOT_INSTALL_JOURNAL"
    fi
)

ensure_sdk_branch() {
    local dest=$1 marker=$2 url=$3 branch=$4 old new listing local_branch=0
    if [[ ! -e "$dest/.git" ]]; then
        echo "Cannot determine the SDK version in $dest. Select a Git checkout of $branch in the config." >&2
        return 1
    fi
    if [[ $(git -C "$dest" branch --show-current) == "$branch" ]]; then return; fi
    old=$(git -C "$dest" rev-parse HEAD)
    if new=$(git -C "$dest" rev-parse --verify --quiet "refs/heads/$branch"); then
        local_branch=1
    else
        listing=$(git ls-remote --exit-code "$url" "refs/heads/$branch") || return 1
        new=${listing%%[[:space:]]*}
    fi
    if [[ "$old" == "$new" ]]; then return; fi
    if [[ -n $(git -C "$dest" status --porcelain) ]]; then
        echo "SDK $dest has local changes. Select a clean checkout for $branch." >&2
        return 1
    fi
    python3 "$(dirname "${BASH_SOURCE[0]}")/_install_packages.py" confirm \
        "Switch SDK $dest: ${old:0:12} -> ${new:0:12} ($branch)" || return 1
    if (( local_branch )); then
        git -C "$dest" cat-file -e "$new:$marker" || return 1
        [[ $(git -C "$dest" rev-parse "refs/heads/$branch") == "$new" ]] || return 1
        git -C "$dest" checkout "$branch" || return 1
    else
        git -C "$dest" fetch --no-tags "$url" "refs/heads/$branch" || return 1
        if [[ $(git -C "$dest" rev-parse FETCH_HEAD) != "$new" ]]; then
            echo "The SDK revision changed after confirmation. Rerun setup to review it." >&2
            return 1
        fi
        git -C "$dest" cat-file -e "$new:$marker" || return 1
        git -C "$dest" checkout --no-track -b "$branch" "$new" || return 1
    fi
    if [[ -n ${REBOT_INSTALL_JOURNAL:-} ]]; then
        printf '[SDK] Updated %s: %s -> %s (%s)\n' "$dest" "${old:0:12}" "${new:0:12}" "$branch" >> "$REBOT_INSTALL_JOURNAL"
    fi
}

clone_sdk() {
    local dest=$1 marker=$2 url=$3
    shift 3
    if [[ -f "$dest/$marker" ]]; then
        if [[ ${1:-} == --branch ]]; then ensure_sdk_branch "$dest" "$marker" "$url" "$2" || return 1; fi
        return 0
    fi
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
    if [[ -n ${REBOT_INSTALL_JOURNAL:-} ]]; then
        printf '[SDK] Installed %s (%s)\n' "$dest" "$(git -C "$dest" rev-parse --short HEAD)" >> "$REBOT_INSTALL_JOURNAL"
    fi
}
