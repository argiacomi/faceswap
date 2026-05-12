#!/usr/bin/env python3
"""Write fixture predictions into the landmark disk cache."""

from __future__ import annotations

import argparse

import numpy as np

from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="outputs/landmark_predictions")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--schema", default="2d_68")
    parser.add_argument("--coordinate-space", default="frame")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args(argv)

    points = np.load(args.prediction).astype("float32")
    prediction = LandmarkPrediction(
        landmarks=points,
        schema=args.schema,
        model_name=args.model,
        source_landmark_count=points.shape[0],
        coordinate_space=args.coordinate_space,
    )
    DiskPredictionCache(args.cache_dir).write(
        args.sample_id,
        prediction,
        checkpoint=args.checkpoint,
        config={"schema": args.schema, "coordinate_space": args.coordinate_space},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
