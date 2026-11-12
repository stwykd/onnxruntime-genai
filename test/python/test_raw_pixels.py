# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest

Image = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def processor():
    model = og.Model(str(Path(__file__).resolve().parents[1] / "models" / "phi3-v"))
    return model.create_multimodal_processor()


def process(processor, images, count=1):
    prompt = "<|user|>\n" + "".join(f"<|image_{i + 1}|>\n" for i in range(count))
    return processor(prompt + "Describe.\n<|end|>\n<|assistant|>\n", images=images)


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def assert_equal(expected, actual):
    assert set(expected.keys()) == set(actual.keys())
    for key in expected.keys():
        left, right = expected[key].as_numpy(), actual[key].as_numpy()
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes(), key


@pytest.mark.parametrize("size", [(1, 1), (17, 33), (33, 17), (31, 31), None])
@pytest.mark.parametrize("format", ["RGB8", "RGBA8", "BGRA8"])
def test_raw_pixels_png_parity(processor, size, format):
    if size is None:
        image = Image.open(Path(__file__).resolve().parents[1] / "images" / "cars.jpg").convert("RGB")
        rgb = np.asarray(image).copy()
    else:
        height, width = size
        rgb = (np.arange(height * width * 3, dtype=np.uint32) % 256).astype(np.uint8).reshape(height, width, 3)
    if format == "RGB8":
        source = rgb
    else:
        alpha = (np.arange(rgb.shape[0] * rgb.shape[1]) % 256).astype(np.uint8).reshape(*rgb.shape[:2], 1)
        source = np.concatenate((rgb, alpha), axis=2)
    encoded = og.Images.open_bytes(png(source))
    if format == "BGRA8":
        source = source[:, :, [2, 1, 0, 3]]
    height, width, channels = source.shape
    padded = np.zeros((height, width + 5, channels), dtype=np.uint8)
    padded[:, :width] = source
    raw = og.Images.from_pixels(padded[:, :width], format=format)
    padded.fill(0)
    expected = process(processor, encoded)
    assert_equal(expected, process(processor, raw))
    assert_equal(expected, process(processor, raw))


@pytest.mark.parametrize("format", ["RGBA8", "BGRA8"])
def test_raw_pixels_premultiplied_alpha(processor, format):
    rgba = np.array([[[17, 18, 19, 0], [64, 32, 0, 128], [1, 0, 1, 1], [255, 128, 1, 255]]], dtype=np.uint8)
    rgb = np.array([[[0, 0, 0], [128, 64, 0], [255, 0, 255], [255, 128, 1]]], dtype=np.uint8)
    source = rgba if format == "RGBA8" else rgba[:, :, [2, 1, 0, 3]].copy()
    raw = og.Images.from_pixels(source, format=format, alpha_mode="premultiplied")
    assert_equal(process(processor, og.Images.open_bytes(png(rgb))), process(processor, raw))
    opaque = og.Images.from_pixels(source, format=format, alpha_mode="opaque")
    assert_equal(process(processor, og.Images.open_bytes(png(rgba[:, :, :3]))), process(processor, opaque))


def test_raw_pixels_batch_order(processor):
    red = np.full((13, 17, 3), [255, 0, 0], dtype=np.uint8)
    blue = np.full((19, 11, 3), [0, 0, 255], dtype=np.uint8)
    raw = og.Images.from_pixels(red, blue)
    encoded = og.Images.open_bytes(png(red), png(blue))
    assert_equal(process(processor, encoded, 2), process(processor, raw, 2))


@pytest.mark.parametrize(
    "array, kwargs",
    [
        (b"pixels", {}),
        (np.zeros((2, 2, 3), dtype=np.float32), {}),
        (np.zeros((2, 2), dtype=np.uint8), {}),
        (np.zeros((0, 2, 3), dtype=np.uint8), {}),
        (np.zeros((2, 2, 4), dtype=np.uint8), {}),
        (np.zeros((2, 2, 3), dtype=np.uint8)[::-1], {}),
        (np.zeros((2, 4, 3), dtype=np.uint8)[:, ::2], {}),
        (np.zeros((2, 2, 3), dtype=np.uint8)[:, :, ::-1], {}),
        (np.zeros((16385, 1, 3), dtype=np.uint8), {}),
        (np.zeros((1, 1, 3), dtype=np.uint8), {"format": "BGR"}),
        (np.zeros((1, 1, 3), dtype=np.uint8), {"alpha_mode": "unknown"}),
        (np.zeros((1, 1, 3), dtype=np.uint8), {"alpha_mode": "premultiplied"}),
    ],
)
def test_raw_pixels_invalid_input(array, kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        og.Images.from_pixels(array, **kwargs)


def test_raw_pixels_empty_batch():
    with pytest.raises((ValueError, RuntimeError)):
        og.Images.from_pixels()
