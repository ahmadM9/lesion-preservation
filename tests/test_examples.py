import numpy as np
from PIL import Image

from lesion_preservation.data import Box
from lesion_preservation.examples import BOX_COLOUR, NORMAL_COLOUR, save_example

SHAPE = (12, 10)
BOX = Box(0, row=4, col=3, height=3, width=2, label="Nonspecific white matter lesion")
SCALE = 3


def saved(tmp_path):
    reference = np.full(SHAPE, 0.5)
    image = np.full(SHAPE, 0.25)
    path = tmp_path / "examples" / "one.png"
    save_example(reference, image, [BOX], 1.0, SCALE, path)
    return np.asarray(Image.open(path))


def test_two_panels_at_scale(tmp_path):
    picture = saved(tmp_path)
    assert picture.shape == (SHAPE[0] * SCALE, 2 * SHAPE[1] * SCALE, 3)


def test_outline_sits_just_outside_the_box(tmp_path):
    picture = saved(tmp_path)
    top, left = BOX.row * SCALE, BOX.col * SCALE
    bottom, right = top + BOX.height * SCALE, left + BOX.width * SCALE
    for offset in (0, SHAPE[1] * SCALE):
        assert tuple(picture[top - 1, left - 1 + offset]) == BOX_COLOUR
        assert tuple(picture[bottom, right + offset]) == BOX_COLOUR
        # the box pixels themselves keep their grey value
        inside = picture[top:bottom, left + offset : right + offset]
        assert (inside == inside[0, 0]).all() and inside[0, 0, 0] == inside[0, 0, 1]


def test_same_grey_scale_for_both_panels(tmp_path):
    picture = saved(tmp_path)
    assert picture[0, 0, 0] == 127
    assert picture[0, SHAPE[1] * SCALE, 0] == 63


def test_normal_box_in_its_own_colour(tmp_path):
    normal = Box(0, row=1, col=6, height=3, width=2, label="")
    path = tmp_path / "normal.png"
    save_example(np.full(SHAPE, 0.5), np.full(SHAPE, 0.25), [BOX], 1.0, SCALE, path, [normal])
    picture = np.asarray(Image.open(path))
    assert tuple(picture[normal.row * SCALE - 1, normal.col * SCALE - 1]) == NORMAL_COLOUR
    assert tuple(picture[BOX.row * SCALE - 1, BOX.col * SCALE - 1]) == BOX_COLOUR
