"""Basic invariants of ``filseg.geometry`` and ``filseg.postprocess`` on synthetic inputs."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from filseg.data import SIZE
from filseg.geometry import Disk, fit_disk_bright, longitude_deg
from filseg.postprocess import PostConfig, components, instances, upsample

DISK = Disk(cx=1024.0, cy=1024.0, r=900.0)


def test_longitude_zero_at_centre_and_ninety_at_limb() -> None:
    lon = longitude_deg(np.array([1024.0, 1924.0, 124.0, 3000.0]), np.array([1024.0] * 4), DISK)
    assert lon[0] == 0.0
    assert np.allclose(lon[1:3], 90.0)
    assert lon[3] == 90.0  # beyond the limb is clipped to 90


def test_longitude_is_symmetric_and_grows_with_distance_from_meridian() -> None:
    xs = np.array([1024.0 + 300, 1024.0 - 300, 1024.0 + 600])
    lon = longitude_deg(xs, np.full(3, 1024.0), DISK)
    assert lon[0] == lon[1] and lon[2] > lon[0]
    assert abs(lon[0] - np.degrees(np.arcsin(300 / 900))) < 1e-9


def test_fit_disk_bright_recovers_a_synthetic_disk() -> None:
    yy, xx = np.mgrid[0:512, 0:512]
    img = (((xx - 250) ** 2 + (yy - 260) ** 2) < 190 ** 2).astype(np.uint8) * 200
    disk = fit_disk_bright(img)
    assert abs(disk.cx - 250 * 4) < 8 and abs(disk.cy - 260 * 4) < 8 and abs(disk.r - 190 * 4) < 8


def make_prob() -> np.ndarray:
    """256 px map: a large blob near the meridian, a large blob near the east limb, a speck."""
    p = np.zeros((256, 256), np.float32)
    p[120:130, 100:160] = 0.9    # large, central
    p[120:130, 230:250] = 0.9    # large, near the limb
    p[20:21, 20:22] = 0.9        # tiny, under 200 px at 2048
    return p


def test_upsample_shape_and_identity() -> None:
    assert upsample(make_prob()).shape == (SIZE, SIZE)
    full = np.ones((SIZE, SIZE), np.float32)
    assert upsample(full) is not None and (upsample(full) == full).all()


def test_components_counts_and_masks_are_disjoint() -> None:
    comps = components(make_prob(), 0.5, 0, DISK)
    assert len(comps) == 3
    assert all(0.5 < c.score <= 0.9 for c in comps)  # bilinear edges pull the mean below 0.9
    assert sum(c.area for c in comps) == sum(int(mask_util.area(c.rle)) for c in comps)
    total = sum(mask_util.decode(c.rle).astype(int) for c in comps)
    assert total.max() == 1


def test_instances_filters_by_area_and_limb() -> None:
    assert len(instances(make_prob(), PostConfig(min_area=200), DISK)) == 2     # the speck is dropped
    assert len(instances(make_prob(), PostConfig(min_area=200, max_lon=70.0), DISK)) == 1  # and the limb blob
    assert instances(make_prob(), PostConfig(min_score=0.95), DISK) == []


def test_instances_of_an_empty_map_is_empty() -> None:
    assert instances(np.zeros((256, 256), np.float32), PostConfig(), DISK) == []
