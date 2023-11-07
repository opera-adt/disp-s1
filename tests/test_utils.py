from disp_s1 import utils


def test_read_zipped_json():
    out = utils.read_zipped_json(utils.FRAME_TO_BURST_JSON_FILE)
    assert sorted(out.keys()) == ["data", "metadata"]
    data_dict = out["data"]
    assert len(data_dict) == 46986
    assert out["metadata"] == {
        "margin": 5000.0,
        "snap": 30.0,
        "target_frame_size": 9,
        "version": "0.1.2",
        "last_modified": "2023-08-04T09:42:56.539031",
        "land_buffer_deg": 0.3,
        "land_optimized": False,
    }


def test_get_frame_json():
    out = utils.get_frame_json(11114)
    assert out == {
        "epsg": 32610.0,
        "is_land": True,
        "is_north_america": True,
        "xmin": 546450,
        "ymin": 4204110,
        "xmax": 833790,
        "ymax": 4409070,
        "burst_ids": [
            "t042_088905_iw1",
            "t042_088905_iw2",
            "t042_088905_iw3",
            "t042_088906_iw1",
            "t042_088906_iw2",
            "t042_088906_iw3",
            "t042_088907_iw1",
            "t042_088907_iw2",
            "t042_088907_iw3",
            "t042_088908_iw1",
            "t042_088908_iw2",
            "t042_088908_iw3",
            "t042_088909_iw1",
            "t042_088909_iw2",
            "t042_088909_iw3",
            "t042_088910_iw1",
            "t042_088910_iw2",
            "t042_088910_iw3",
            "t042_088911_iw1",
            "t042_088911_iw2",
            "t042_088911_iw3",
            "t042_088912_iw1",
            "t042_088912_iw2",
            "t042_088912_iw3",
            "t042_088913_iw1",
            "t042_088913_iw2",
            "t042_088913_iw3",
        ],
    }


def test_get_frame_bbox():
    epsg, bbox = utils.get_frame_bbox(11114)
    assert epsg == 32610
    assert bbox == [546450, 4204110, 833790, 4409070]
