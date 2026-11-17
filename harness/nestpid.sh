#!/bin/bash
# CLAUDE_CODE_SHELL_PREFIX for writer sandboxes: each Bash tool command runs in its own pid
# namespace, so it cannot reach the claude process (e.g. /proc/<pid>/fd/1, the transcript stream).
exec /usr/bin/bwrap --die-with-parent --unshare-pid --dev-bind / / --proc /proc -- /bin/bash -c "$*"
