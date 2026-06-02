#!/usr/bin/env python3
"""Production landmark-ensemble artifact bundle: install, locate, load.

Background
----------
The landmark resolver pipeline produces a tree of intermediate artifacts
under ``outputs/landmarks/run_*/resolver_pipeline/``. Only a small subset
of that tree is needed at extract time: the promoted setup file (which
itself references its weights file relatively), and one scorer artifact
per supported runtime policy.

The runtime used to read each artifact path directly out of
``extract.ini`` (``setup_path``, ``weights_path``, ``resolver_scorer_path``).
That made paths user-settable production knobs that could fall out of
sync with the artifact contents — e.g. a learned resolver policy pointing
at the wrong LightGBM scorer JSON. The runtime loader picks the scorer
implementation from the artifact's ``model_type``, not from the policy
name, so the mismatch only surfaced at predict time.

This module replaces that contract. The pipeline installs a *production
bundle* into a known location, and runtime resolves artifacts from that
bundle by policy. Config now only carries behavior (``resolver_policy``,
``use_alignment_resolver``, ...), not paths.

Bundle layout
-------------
Default location: ``<project_root>/.fs_cache/landmark_ensemble/current/``.
Override with the ``FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS`` env var (point
it at an alternate ``current/`` directory).

::

    current/
        manifest.json
        best_setup.json
        best_weights.json
        scorers/
            learned_quality_v3.json

Manifest schema::

    {
        "artifact_schema_version": 1,
        "active_policy": "learned_quality_v3",
        "setup": "best_setup.json",
        "scorers": {
            "learned_quality_v3": "scorers/learned_quality_v3.json"
        },
        "created_by": "run_landmark_resolver_pipeline.py",
        "source_output_root": "outputs/landmarks/run_.../resolver_pipeline"
    }

The manifest deliberately does **not** carry a ``weights`` key.
``best_setup.json`` already references ``best_weights.json`` via a
relative ``weights_path``, and ``load_promoted_setup`` resolves it
against the setup file's directory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import typing as T
from dataclasses import dataclass
from pathlib import Path

from lib.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1

DEFAULT_BUNDLE_DIR = Path(PROJECT_ROOT) / ".fs_cache" / "landmark_ensemble" / "current"
"""Default install location for the production bundle.

``.fs_cache`` is already gitignored and is the project's "binary/runtime
artifacts live here" location, so the bundle lands alongside model caches
rather than next to source-controlled config templates.
"""

BUNDLE_DIR_ENV = "FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS"
"""Environment variable that overrides the bundle install location.

Point it at an alternate ``current/`` directory — useful for CI, A/B
testing two pipelines side-by-side, and tests that need to install into
a tmp_path.
"""

MANIFEST_FILENAME = "manifest.json"
SETUP_FILENAME = "best_setup.json"
WEIGHTS_FILENAME = "best_weights.json"
SCORERS_SUBDIR = "scorers"

LEARNED_POLICIES: tuple[str, ...] = ("learned_quality_v3",)
"""Resolver policies that consume a scorer artifact.

``roll_aware_veto`` is intentionally absent — its runtime path runs
priority/veto resolution without a learned scorer.
"""

ROLL_AWARE_VETO_POLICY = "roll_aware_veto"


class ProductionBundleError(RuntimeError):
    """Base class for production bundle errors."""


class ProductionBundleMissing(ProductionBundleError):
    """No production bundle is installed at the resolved bundle directory."""


class ProductionBundleInvalid(ProductionBundleError):
    """A bundle exists but its manifest or referenced files are unusable."""


def bundle_dir() -> Path:
    """Return the resolved production bundle directory.

    Honors ``FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS`` if set, otherwise
    falls back to ``<project_root>/.fs_cache/landmark_ensemble/current``.
    """
    override = os.environ.get(BUNDLE_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_BUNDLE_DIR


def manifest_path() -> Path:
    """Return the resolved manifest path."""
    return bundle_dir() / MANIFEST_FILENAME


@dataclass(frozen=True)
class ProductionBundle:
    """In-memory view of an installed production bundle."""

    bundle_dir: Path
    setup_path: Path
    scorers: T.Mapping[str, Path]
    active_policy: str
    manifest: T.Mapping[str, T.Any]

    def scorer_path_for(self, policy: str) -> Path | None:
        """Return the scorer artifact path for ``policy``.

        ``roll_aware_veto`` returns ``None`` (the runtime resolver does
        not need a scorer for that path). A configured learned policy
        with no installed scorer raises ``ProductionBundleInvalid`` —
        config asked for a learned policy but the bundle was built
        without that scorer version.
        """
        if policy == ROLL_AWARE_VETO_POLICY:
            return None
        if policy not in LEARNED_POLICIES:
            raise ProductionBundleInvalid(
                f"unsupported resolver_policy {policy!r}; expected one of "
                f"{(ROLL_AWARE_VETO_POLICY, *LEARNED_POLICIES)}"
            )
        path = self.scorers.get(policy)
        if path is None:
            raise ProductionBundleInvalid(
                f"production bundle at {self.bundle_dir} has no installed scorer for "
                f"resolver_policy={policy!r}; available scorers: {sorted(self.scorers)}"
            )
        return path


def load_production_bundle() -> ProductionBundle:
    """Read and validate the installed production bundle.

    Raises ``ProductionBundleMissing`` when no manifest is present, and
    ``ProductionBundleInvalid`` when the manifest exists but its
    schema, referenced setup, or referenced scorers are unusable.
    """
    target = bundle_dir()
    manifest_file = target / MANIFEST_FILENAME
    if not manifest_file.is_file():
        raise ProductionBundleMissing(
            f"no production landmark-ensemble bundle installed at {target} "
            f"(expected manifest at {manifest_file}). Run "
            "`tools/landmarks/run_landmark_resolver_pipeline.py ... --write-config` "
            f"or set {BUNDLE_DIR_ENV} to an alternate bundle directory."
        )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ProductionBundleInvalid(
            f"failed to read production bundle manifest at {manifest_file}: {err}"
        ) from err
    if not isinstance(manifest, dict):
        raise ProductionBundleInvalid(
            f"production bundle manifest at {manifest_file} must be a JSON object"
        )
    schema = int(manifest.get("artifact_schema_version", 0))
    if schema != ARTIFACT_SCHEMA_VERSION:
        raise ProductionBundleInvalid(
            f"production bundle manifest at {manifest_file} has "
            f"artifact_schema_version={schema}, expected {ARTIFACT_SCHEMA_VERSION}"
        )
    setup_rel = manifest.get("setup")
    if not isinstance(setup_rel, str) or not setup_rel:
        raise ProductionBundleInvalid(
            f"production bundle manifest at {manifest_file} is missing 'setup'"
        )
    setup_file = (target / setup_rel).resolve()
    if not setup_file.is_file():
        raise ProductionBundleInvalid(f"production bundle setup file is missing: {setup_file}")
    raw_scorers = manifest.get("scorers", {})
    if not isinstance(raw_scorers, dict):
        raise ProductionBundleInvalid(
            f"production bundle manifest at {manifest_file} has non-dict 'scorers'"
        )
    scorers: dict[str, Path] = {}
    for policy, rel in raw_scorers.items():
        if policy not in LEARNED_POLICIES:
            # Tolerate unknown policies in the manifest so the bundle
            # format can carry new policies before the runtime knows
            # about them; just don't expose them via scorer_path_for.
            logger.debug(
                "[ProductionBundle] manifest at %s lists unknown policy %r — ignoring",
                manifest_file,
                policy,
            )
            continue
        if not isinstance(rel, str) or not rel:
            raise ProductionBundleInvalid(
                f"production bundle manifest at {manifest_file} has invalid path for "
                f"scorer policy {policy!r}"
            )
        scorer_file = (target / rel).resolve()
        if not scorer_file.is_file():
            raise ProductionBundleInvalid(
                f"production bundle scorer for policy {policy!r} is missing: {scorer_file}"
            )
        scorers[policy] = scorer_file
    active_policy = str(manifest.get("active_policy") or "")
    return ProductionBundle(
        bundle_dir=target,
        setup_path=setup_file,
        scorers=scorers,
        active_policy=active_policy,
        manifest=manifest,
    )


def install_production_bundle(
    *,
    setup_src: Path,
    weights_src: Path,
    scorer_sources: T.Mapping[str, Path],
    active_policy: str,
    source_output_root: Path | None = None,
    created_by: str = "run_landmark_resolver_pipeline.py",
    dest: Path | None = None,
) -> Path:
    """Install a production bundle from pipeline source artifacts.

    Copies ``setup_src`` and ``weights_src`` (the setup file references
    weights relatively, so both must land in the same directory) plus
    each present entry in ``scorer_sources`` (a policy → source-path
    map). Writes a manifest.json atomically and returns its path.

    Only policies in ``LEARNED_POLICIES`` are installed; unknown keys in
    ``scorer_sources`` raise ``ValueError`` so the pipeline can't silently
    install a scorer the runtime doesn't know how to use.

    ``dest`` defaults to ``bundle_dir()`` so callers don't have to thread
    the env override through; pass an explicit path in tests.
    """
    if active_policy not in (ROLL_AWARE_VETO_POLICY, *LEARNED_POLICIES):
        raise ValueError(f"active_policy {active_policy!r} is not a supported resolver policy")
    unknown = sorted(set(scorer_sources) - set(LEARNED_POLICIES))
    if unknown:
        raise ValueError(
            f"scorer_sources contains unsupported policies: {unknown}; "
            f"expected subset of {LEARNED_POLICIES}"
        )
    target = (dest or bundle_dir()).resolve()
    target.mkdir(parents=True, exist_ok=True)
    scorers_dir = target / SCORERS_SUBDIR
    scorers_dir.mkdir(parents=True, exist_ok=True)

    setup_dst = target / SETUP_FILENAME
    weights_dst = target / WEIGHTS_FILENAME
    shutil.copyfile(setup_src, setup_dst)
    shutil.copyfile(weights_src, weights_dst)

    scorer_manifest: dict[str, str] = {}
    for policy in LEARNED_POLICIES:
        source = scorer_sources.get(policy)
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            logger.debug(
                "[ProductionBundle] skipping scorer for %s — source missing: %s",
                policy,
                source_path,
            )
            continue
        dst_rel = f"{SCORERS_SUBDIR}/{policy}.json"
        shutil.copyfile(source_path, target / dst_rel)
        scorer_manifest[policy] = dst_rel

    if active_policy in LEARNED_POLICIES and active_policy not in scorer_manifest:
        raise ValueError(
            f"active_policy={active_policy!r} but no scorer was installed for it; "
            f"scorer_sources={sorted(scorer_sources)}"
        )

    manifest_payload: dict[str, T.Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "active_policy": active_policy,
        "setup": SETUP_FILENAME,
        "scorers": scorer_manifest,
        "created_by": created_by,
    }
    if source_output_root is not None:
        manifest_payload["source_output_root"] = str(source_output_root)

    manifest_file = target / MANIFEST_FILENAME
    tmp_file = manifest_file.with_suffix(".json.tmp")
    tmp_file.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_file, manifest_file)
    logger.info(
        "[ProductionBundle] installed at %s active_policy=%s scorers=%s",
        target,
        active_policy,
        sorted(scorer_manifest),
    )
    return manifest_file


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BUNDLE_DIR_ENV",
    "DEFAULT_BUNDLE_DIR",
    "LEARNED_POLICIES",
    "MANIFEST_FILENAME",
    "ROLL_AWARE_VETO_POLICY",
    "SCORERS_SUBDIR",
    "SETUP_FILENAME",
    "WEIGHTS_FILENAME",
    "ProductionBundle",
    "ProductionBundleError",
    "ProductionBundleInvalid",
    "ProductionBundleMissing",
    "bundle_dir",
    "install_production_bundle",
    "load_production_bundle",
    "manifest_path",
]
