"""TuneOS brand component library.

Logo, loaders, skeletons, and liquid progress — all driven by the metaball
identity. See docs/brand/BRAND_SPEC.md. Regenerate the SVG assets with
scripts/build_brand_assets.py.
"""

from app.components.brand.loader import (
    metaball_loader,
    metaball_overlay,
    metaball_spinner,
)
from app.components.brand.logo import tune_lockup, tune_mark, tune_wordmark
from app.components.brand.progress import (
    liquid_progress,
    liquid_progress_indeterminate,
)
from app.components.brand.skeleton import (
    skeleton_block,
    skeleton_card,
    skeleton_circle,
    skeleton_list,
    skeleton_table,
    skeleton_text,
)

__all__ = [
    "metaball_loader",
    "metaball_overlay",
    "metaball_spinner",
    "tune_mark",
    "tune_wordmark",
    "tune_lockup",
    "liquid_progress",
    "liquid_progress_indeterminate",
    "skeleton_block",
    "skeleton_card",
    "skeleton_circle",
    "skeleton_list",
    "skeleton_table",
    "skeleton_text",
]
