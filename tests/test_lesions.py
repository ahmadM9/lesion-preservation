import pytest

from lesion_preservation.config import LesionsConfig
from lesion_preservation.data import Box
from lesion_preservation.lesions import boxes_from_labels, join, lesion_table, size_bin

CFG = LesionsConfig(
    label="Nonspecific white matter lesion",
    contrast="AXFLAIR",
    connectivity=26,
    connectivity_variant=6,
    size_upper_bounds=(7, 16),
    large_box_px=51,
)
FILE = "file_brain_AXFLAIR_000_0000000"


def box(box_id, row, col, height=4, width=4):
    return Box(box_id, row, col, height, width, CFG.label)


@pytest.mark.parametrize("connectivity", [6, 26])
def test_overlap_on_neighbouring_slices_joins(connectivity):
    items = [(FILE, 3, box(0, 10, 10)), (FILE, 4, box(1, 13, 13))]
    assert join(items, connectivity) == {0: 0, 1: 0}


@pytest.mark.parametrize("connectivity", [6, 26])
def test_chain_over_three_slices_is_one_lesion(connectivity):
    # slice 3 and slice 5 do not overlap each other; slice 4 links them
    items = [(FILE, 3, box(5, 10, 10)), (FILE, 4, box(2, 10, 13)), (FILE, 5, box(9, 10, 16))]
    assert join(items, connectivity) == {5: 2, 2: 2, 9: 2}


@pytest.mark.parametrize("connectivity", [6, 26])
def test_overlap_on_same_slice_joins(connectivity):
    items = [(FILE, 3, box(0, 10, 10)), (FILE, 3, box(1, 11, 11))]
    assert join(items, connectivity) == {0: 0, 1: 0}


@pytest.mark.parametrize("connectivity", [6, 26])
def test_two_slices_apart_is_not_joined(connectivity):
    items = [(FILE, 3, box(0, 10, 10)), (FILE, 5, box(1, 10, 10))]
    assert join(items, connectivity) == {0: 0, 1: 1}


@pytest.mark.parametrize("connectivity", [6, 26])
def test_different_files_are_not_joined(connectivity):
    items = [(FILE, 3, box(0, 10, 10)), ("file_brain_AXFLAIR_000_0000001", 4, box(1, 10, 10))]
    assert join(items, connectivity) == {0: 0, 1: 1}


@pytest.mark.parametrize("connectivity, joined", [(6, False), (26, True)])
def test_corner_contact_on_neighbouring_slices(connectivity, joined):
    # rows 10 to 13 then 14 to 17, cols 10 to 13 then 14 to 17: only corners touch
    items = [(FILE, 3, box(0, 10, 10)), (FILE, 4, box(1, 14, 14))]
    assert (join(items, connectivity)[1] == 0) is joined


@pytest.mark.parametrize("connectivity, joined", [(6, True), (26, True)])
def test_side_contact_on_same_slice(connectivity, joined):
    # cols 10 to 13 then 14 to 17: neighbouring pixels share a face
    items = [(FILE, 3, box(0, 10, 10)), (FILE, 3, box(1, 10, 14))]
    assert (join(items, connectivity)[1] == 0) is joined


@pytest.mark.parametrize(
    "size, name", [(1, "<=7"), (7, "<=7"), (8, "8-16"), (16, "8-16"), (17, ">16")]
)
def test_size_bin_edges(size, name):
    assert size_bin(size, CFG.size_upper_bounds) == name


def test_lesion_size_is_longest_side_across_boxes():
    items = [(FILE, 3, box(4, 10, 10, 3, 5)), (FILE, 4, box(7, 10, 10, 9, 2))]
    (lesion,) = lesion_table(items, CFG, CFG.connectivity)
    assert (lesion.lesion_id, lesion.slices, lesion.num_boxes) == (4, (3, 4), 2)
    assert (lesion.size, lesion.size_bin, lesion.large) == (9, "8-16", False)


@pytest.mark.parametrize("side, large", [(51, False), (52, True)])
def test_large_flag(side, large):
    (lesion,) = lesion_table([(FILE, 0, box(0, 0, 0, side, 2))], CFG, CFG.connectivity)
    assert lesion.large is large


def test_boxes_from_labels_filters_label_contrast_and_study_level(tmp_path):
    csv_path = tmp_path / "brain.csv"
    csv_path.write_text(
        "file,slice,study_level,x,y,width,height,label\n"
        f"{FILE},1,No,5,6,3,4,Nonspecific white matter lesion\n"
        f"{FILE},1,No,5,6,3,4,Mass\n"
        f"{FILE},0,Yes,,,,,Normal for age\n"
        "file_brain_AXT1_000_0000000,1,No,5,6,3,4,Nonspecific white matter lesion\n"
    )
    boxes = boxes_from_labels(csv_path, CFG.label, CFG.contrast)
    assert boxes == [(FILE, 1, Box(0, 6, 5, 4, 3, CFG.label))]
