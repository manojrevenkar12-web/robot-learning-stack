"""Add the per-episode scene layout (``episode_spec``) to an existing release's meta/rls_episodes.parquet.

Newer releases write this column directly (rls.data.export_lerobot); this backfills older ones
without re-encoding any video.

    python scripts/add_scene_metadata.py data/release/rls_panda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from rls.data.schema import read_episode


def main() -> None:
    root = Path(sys.argv[1])
    p = root / "meta" / "rls_episodes.parquet"
    side = pd.read_parquet(p)
    side["episode_spec"] = [json.dumps(read_episode(Path(sp), images=False).episode_spec, default=str) for sp in side["staging_path"]]
    side.to_parquet(p, index=False)
    print(f"{len(side)} episodes updated: {p}")


if __name__ == "__main__":
    main()
