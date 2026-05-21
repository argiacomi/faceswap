"""Landmark-driven alignment decisioning for extract.

This subsystem owns runtime-facing alignment choices that aren't just
ensemble fusion — geometry-risk routing, fallback selection, future
adaptive resolvers. Ensemble weighting still lives in
:mod:`lib.landmarks.ensemble`; the resolver consumes those primitives
but operates one layer up.
"""
