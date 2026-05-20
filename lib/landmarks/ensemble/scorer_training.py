#!/usr/bin/env python3
"""Library-facing runtime resolver scorer training implementation.

Implementation is temporarily re-exported from the legacy tools module while
imports migrate. Once all callers use this module and CI passes, the legacy
module can become a thin shim or be deleted with the remaining merge-candidate
cleanup.
"""

from tools.landmarks.scorer_training import *  # noqa: F401,F403
