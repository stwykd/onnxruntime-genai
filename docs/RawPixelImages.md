# Decoded image input

This branch pins the matching Extensions implementation on the contributor's fork.
For upstream integration, merge Extensions first and update the dependency to its
upstream commit.

`OgaLoadImagesFromRawPixels` loads decoded, top-down uint8 pixels without a PNG or
JPEG encode/decode round trip. `OgaLoadImagesFromBuffers` continues to take encoded
bytes. Both return `OgaImages` and use the same multimodal processor.

```cpp
OgaRawImage image{
    pixels, buffer_size, width, height, row_bytes,
    OgaPixelFormatBGRA8, OgaAlphaStraight};
auto images = OgaImages::LoadFromRawPixels(&image, 1);
auto inputs = processor->ProcessImages(prompt, images.get());
```

In C, pass an array of `OgaRawImage` to
`OgaLoadImagesFromRawPixels(descriptors, count, &images)` and destroy the result with
`OgaDestroyImages`. All buffers are copied before returning.

```python
images = og.Images.from_pixels(rgb_uint8_array)
images = og.Images.from_pixels(bgra1, bgra2, format="BGRA8", alpha_mode="straight")
inputs = processor(prompt, images=images)
```

Python arrays have shape `(height, width, channels)`, dtype `uint8`, packed channels
and pixels, and a positive row stride. Row padding is supported; negative strides,
transposes and subsampled columns are rejected rather than implicitly copied.

Supported formats are RGB8, RGBA8 and BGRA8, in byte order (not a platform-native
32-bit integer order). Opaque and straight alpha are discarded without compositing,
matching the built-in PNG decoder. Premultiplied RGBA/BGRA is explicitly supported:
each channel becomes `min(255, (channel * 255 + alpha / 2) / alpha)`; alpha zero
becomes black. RGB8 cannot be premultiplied. This cannot recover precision already
lost to premultiplication and is not guaranteed to match every image encoder's
unpremultiplication rounding. For exact parity with an application's PNG encoder,
provide the same straight RGB/RGBA values it would encode.

Dimensions are limited to 16384 per axis and 100 million pixels per image, matching
the built-in decoder limits. A batch can own at most 512 MiB of RGB pixels.
`data_size` must cover `(height - 1) * row_stride + width * channels`, checked for
overflow before reading. An invalid batch fails atomically.

The processor configuration must start with `DecodeImage`. For raw input that
operation copies RGB pixels instead of invoking a codec, respecting `color_space`;
all remaining transforms and model-specific prompt handling are unchanged. Engine
inputs retain width/height metadata for admission checks. No color management, EXIF
rotation, GPU readback, or alpha compositing is performed. Callers must do those
before loading when needed. This is a copying API, not zero-copy.

Processor creation still initializes configured codecs. A Windows sandbox that
blocks WIC must build Extensions with `OCOS_ENABLE_VENDOR_IMAGE_CODECS=OFF`, including
when using raw images.
