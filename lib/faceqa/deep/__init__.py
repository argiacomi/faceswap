#!/usr/bin/env python3
"""FaceQA deep-analysis (DECA) audit package.

This package adds the optional ``--deep-analysis deca`` mode to the FaceQA
coverage workflow (issue #156). It is intentionally isolated from the default
lightweight coverage path: nothing here is imported unless deep analysis is
requested, so the standard ``faceqa --mode coverage`` run never pays the
``torch`` / DECA import or model-load cost.

Layout
------
* :mod:`lib.faceqa.deep.metrics` - pure-numpy coverage metrics over DECA
  coefficient vectors (expression-space, lighting-space, latent entropy,
  cluster coverage). No heavy dependencies; fully unit-testable.
* :mod:`lib.faceqa.deep.deca_encoder` - the torch DECA encoder that maps an
  aligned face crop to FLAME pose / expression / lighting coefficients.
* :mod:`lib.faceqa.deep.weights` - standard model-cache download + loader for
  the DECA weights.
* :mod:`lib.faceqa.deep.audit` - orchestration: reconstruct crops, run the
  encoder, aggregate coefficients, compute metrics, and build the deep-audit
  report block (including the landmark-vs-DECA comparison).
"""

from lib.utils import get_module_objects

__all__ = get_module_objects(__name__)
