"""Small NumPy RGB resampler candidate; four-case qualification is separate.

Extracted from retained prepare.py SHA256 e5dd5759e142588d6b3db7b55769d13248598dd1cca69baffe700b8ce7ca3c64.
Arithmetic follows PyTorch 5c4886908584029761b579af026dcfb627c84070
aten/src/ATen/native/cpu/UpSampleKernel.cpp (PyTorch BSD-style license).
No Torch import, normalization, tokenization, model or request handling.
"""
import math


def _cubic_aa(value):
    # PyTorch 5c488690: Keys cubic, antialias=True; keep binary64 operation order.
    x, a = abs(value), -0.5
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


def _uint8_coefficients(input_size, output_size):
    """Antialiased bicubic binary64 weights and int16 quantization, align_corners=False."""
    scale = input_size / output_size
    support = 2.0 * scale if scale >= 1.0 else 2.0
    maximum = math.ceil(support) * 2 + 1
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    rows, largest = [], 0.0
    for i in range(output_size):
        center = scale * (i + 0.5)
        first = max(int(center - support + 0.5), 0)
        count = min(max(min(int(center + support + 0.5), input_size) - first, 0), maximum)
        weights, total = [], 0.0
        for j in range(count):
            weight = _cubic_aa((j + first - center + 0.5) * invscale)
            weights.append(weight)
            total += weight
        if total != 0.0:
            for j in range(count):
                weights[j] /= total
                largest = max(largest, weights[j])
        rows.append((first, weights))
    precision = 0
    while precision < 22:
        if int(0.5 + largest * (1 << (precision + 1))) >= 1 << 15:
            break
        precision += 1
    quantized = []
    for first, weights in rows:
        row = []
        for weight in weights:
            value = weight * (1 << precision)
            row.append(int(value - 0.5) if value < 0 else int(value + 0.5))
        quantized.append((first, tuple(row)))
    return tuple(quantized), precision


def resize_uint8_rgb(np, image, size):
    """Horizontal/vertical integer passes; size is (height, width), input is RGB."""
    pixels = np.asarray(image, dtype=np.uint8)
    height, width = pixels.shape[:2]
    out_height, out_width = size
    if out_width != width:
        coefficients, precision = _uint8_coefficients(width, out_width)
        horizontal = np.empty((height, out_width, 3), dtype=np.uint8)
        for x, (first, weights) in enumerate(coefficients):
            value = np.full((height, 3), 1 << (precision - 1), dtype=np.int32)
            for offset, weight in enumerate(weights):
                value += np.multiply(pixels[:, first + offset, :], weight, dtype=np.int32)
            horizontal[:, x, :] = np.clip(value >> precision, 0, 255).astype(np.uint8)
        pixels = horizontal
    if out_height != height:
        coefficients, precision = _uint8_coefficients(height, out_height)
        vertical = np.empty((out_height, out_width, 3), dtype=np.uint8)
        for y, (first, weights) in enumerate(coefficients):
            value = np.full((out_width, 3), 1 << (precision - 1), dtype=np.int32)
            for offset, weight in enumerate(weights):
                value += np.multiply(pixels[first + offset, :, :], weight, dtype=np.int32)
            vertical[y, :, :] = np.clip(value >> precision, 0, 255).astype(np.uint8)
        pixels = vertical
    return pixels
