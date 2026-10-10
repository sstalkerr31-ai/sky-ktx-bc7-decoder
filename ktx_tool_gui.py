#!/usr/bin/env python3
"""
KTX Texture Tool - простое GUI для личного использования
============================================================
Decode:  выбрал .ktx -> увидел превью -> сам решил, сохранять ли PNG и куда.
Encode:  выбрал новую картинку + оригинальный .ktx (для формата/заголовка)
         -> (по желанию растянул до 2K/4K/8K) -> Save As.
Upscale: выбрал .ktx -> декод -> растяжение до 8K -> энкод -> Save As.

Массовые вкладки (работают сразу с папками игры, по файлам параллельно):
  * Массовый Decode: две папки с .ktx (A и B) -> все текстуры в PNG.
  * Массовый Encode, три режима:
      1) перекодировать ВСЕ .ktx в выбранное разрешение (декод -> растяжение
         -> энкод) и записать обратно в папки игры;
      2) одна картинка -> во все .ktx (формат/слои/мипы берутся у каждого
         оригинала);
      3) режим изучения: на каждую текстуру рисуется её номер, кодируется в
         .ktx, а в лог пишется, какая текстура какой номер получила.
    Оригиналы перед перезаписью копируются в папку бэкапа (один раз - повторный
    запуск не затирает чистый оригинал), кнопка "Восстановить" возвращает их.

Поддерживается: KTX v1, BC7 (sRGB/linear), BC4, RGBA8 (обычный и sRGB),
простые 2D-текстуры и массивы слоёв. У массива слои в PNG лежат рядом
слева направо (2 слоя 256x256 -> картинка 512x256); при Encode подавай
картинку в таком же виде.

Что нового в этой версии:
  * кодирование переписано на NumPy и разбито на полосы, которые считаются
    параллельно на всех ядрах (отдельные процессы) - 8K занимает секунды;
  * декодирование BC7/BC4 идёт через встроенный в Pillow C-декодер
    (точный для всех режимов BC7, а не усреднённый), старый питоновский
    декодер оставлен как запасной;
  * кодируется тот формат, который указан в оригинальном .ktx
    (BC7 / BC4 / RGBA8), а не всегда BC7;
  * если в оригинале были мип-уровни - они генерируются заново;
  * интерфейс не зависает: работа идёт в фоне, есть прогресс-бар.

Требования: Python 3.8+, Pillow, NumPy
    pip install Pillow numpy

Запуск: python ktx_tool_gui.py
"""
import colorsys
import csv
import functools
import json
import math
import multiprocessing
import os
import queue
import shutil
import struct
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

Image.MAX_IMAGE_PIXELS = None  # 8K-картинки - это нормально, не "бомба"

_RES = getattr(Image, "Resampling", Image)

SIZE_CHOICES = {
    "Как есть": None,
    "2K (2048)": 2048,
    "4K (4096)": 4096,
    "8K (8192)": 8192,
}
FILTER_CHOICES = {
    "Lanczos (плавно)": _RES.LANCZOS,
    "Bicubic": _RES.BICUBIC,
    "Nearest (резкие пиксели)": _RES.NEAREST,
}


# ============================================================
#  KTX: разбор
# ============================================================

KTX_IDENTIFIER = bytes([0xAB, ord('K'), ord('T'), ord('X'), ord(' '),
                        ord('1'), ord('1'), 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])

HEADER_NAMES = ['glType', 'glTypeSize', 'glFormat', 'glInternalFormat',
                'glBaseInternalFormat', 'pixelWidth', 'pixelHeight', 'pixelDepth',
                'numberOfArrayElements', 'numberOfFaces', 'numberOfMipmapLevels',
                'bytesOfKeyValueData']

# glInternalFormat -> (имя, тип кодека)
FORMATS = {
    0x8E8D: ("BC7 sRGB", "bc7"),
    0x8E8C: ("BC7 linear", "bc7"),
    0x8DBB: ("BC4 / RGTC1", "bc4"),
    0x8058: ("RGBA8 (несжатый)", "rgba8"),
    0x8C43: ("RGBA8 sRGB (несжатый)", "rgba8"),
}


def parse_ktx_bytes(data: bytes):
    """Возвращает (header, данные mip0, длина заголовка вместе с key/value)."""
    if data[0:12] != KTX_IDENTIFIER:
        raise ValueError("Не похоже на KTX v1 файл (неверная сигнатура)")
    if struct.unpack_from("<I", data, 12)[0] != 0x04030201:
        raise ValueError("KTX с обратным порядком байт (big-endian) не поддерживается")
    header = dict(zip(HEADER_NAMES, struct.unpack_from("<12I", data, 16)))
    if header['glInternalFormat'] not in FORMATS:
        raise ValueError(f"Формат 0x{header['glInternalFormat']:X} не поддерживается")
    header_len = 16 + 48 + header['bytesOfKeyValueData']
    image_size = struct.unpack_from("<I", data, header_len)[0]
    image_data = data[header_len + 4:header_len + 4 + image_size]
    return header, image_data, header_len


def parse_ktx(path):
    with open(path, "rb") as f:
        data = f.read()
    header, image_data, _ = parse_ktx_bytes(data)
    return header, image_data


def check_simple_2d(header):
    if header['numberOfFaces'] > 1:
        raise ValueError("Этот .ktx содержит кубмапу.\n"
                         "Инструмент поддерживает только 2D-текстуры и массивы слоёв.")
    if header['pixelDepth'] > 1:
        raise ValueError("Этот .ktx содержит 3D-текстуру - не поддерживается.")


def layer_count(header):
    """Сколько слоёв в текстуре (1 для обычной 2D)."""
    return max(1, header['numberOfArrayElements'])


def ktx_layer_count(path):
    with open(path, "rb") as f:
        data = f.read(64)
    if data[0:12] != KTX_IDENTIFIER:
        raise ValueError("Не похоже на KTX v1 файл (неверная сигнатура)")
    return layer_count(dict(zip(HEADER_NAMES, struct.unpack_from("<12I", data, 16))))


# ============================================================
#  Декодеры
# ============================================================

def _pil_bcn_decode(image_data, width, height, n):
    """Быстрый точный декодер BC4/BC7, встроенный в Pillow (написан на C)."""
    bw, bh = (width + 3) // 4, (height + 3) // 4
    need = bw * bh * (16 if n == 7 else 8)
    data = bytes(image_data[:need])
    if len(data) < need:
        data += b"\x00" * (need - len(data))
    mode = "RGBA" if n == 7 else "L"
    img = Image.frombytes(mode, (bw * 4, bh * 4), data, "bcn", n)
    return img.crop((0, 0, width, height)).convert("RGBA")


# ---------- запасной питоновский декодер BC7 (медленный, мульти-сабсеты приближённо) ----------

WEIGHTS = {
    2: [0, 21, 43, 64],
    3: [0, 9, 18, 27, 37, 46, 55, 64],
    4: [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64],
}

MODE_PARAMS = {
    0: dict(NS=3, PB=4, RB=0, ISB=0, CB=4, AB=0, EPB=1, SPB=0, IB=3),
    1: dict(NS=2, PB=6, RB=0, ISB=0, CB=6, AB=0, EPB=0, SPB=1, IB=3),
    2: dict(NS=3, PB=6, RB=0, ISB=0, CB=5, AB=0, EPB=0, SPB=0, IB=2),
    3: dict(NS=2, PB=6, RB=0, ISB=0, CB=7, AB=0, EPB=1, SPB=0, IB=2),
    4: dict(NS=1, PB=0, RB=2, ISB=1, CB=5, AB=6, EPB=0, SPB=0, IB=2, IB2=3),
    5: dict(NS=1, PB=0, RB=2, ISB=0, CB=7, AB=8, EPB=0, SPB=0, IB=2, IB2=2),
    6: dict(NS=1, PB=0, RB=0, ISB=0, CB=7, AB=7, EPB=1, SPB=0, IB=4),
    7: dict(NS=2, PB=6, RB=0, ISB=0, CB=5, AB=5, EPB=1, SPB=0, IB=2),
}


def expand_bits(val, bits):
    if bits >= 8:
        return val & 0xFF
    val <<= (8 - bits)
    return (val | (val >> bits)) & 0xFF


def interpolate(e0, e1, weight):
    return tuple(((64 - weight) * a + weight * b + 32) >> 6 for a, b in zip(e0, e1))


def apply_rotation(rgba, rot):
    r, g, b, a = rgba
    if rot == 1:
        return (a, g, b, r)
    if rot == 2:
        return (r, a, b, g)
    if rot == 3:
        return (r, g, a, b)
    return (r, g, b, a)


class BitReader:
    def __init__(self, block16):
        self.v = int.from_bytes(block16, "little")
        self.pos = 0

    def read(self, n):
        if n == 0:
            return 0
        val = (self.v >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return val


def decode_block_exact_ns1(br, params):
    rot = br.read(params["RB"]) if params["RB"] else 0
    isb = br.read(1) if params["ISB"] else 0
    CB, AB = params["CB"], params["AB"]
    comps_raw = [[br.read(CB) for _ in range(2)] for _ in range(3)]
    if AB > 0:
        comps_raw.append([br.read(AB) for _ in range(2)])
    else:
        comps_raw.append([255, 255])
    pbits = [0, 0]
    if params["EPB"]:
        pbits = [br.read(1), br.read(1)]

    def endpoint(e_idx):
        out = []
        for ci, raw_pair in enumerate(comps_raw):
            bits = CB if ci < 3 else AB
            v = raw_pair[e_idx]
            if params["EPB"]:
                v = (v << 1) | pbits[e_idx]
                bits += 1
            out.append(expand_bits(v, bits) if bits < 8 else (v & 0xFF))
        return tuple(out)

    e0, e1 = endpoint(0), endpoint(1)
    IB = params["IB"]
    IB2 = params.get("IB2")
    pixels = [None] * 16

    if IB2 is None:
        for i in range(16):
            nb = IB if i != 0 else IB - 1
            idx = br.read(nb)
            pixels[i] = interpolate(e0, e1, WEIGHTS[IB][idx])
    else:
        idx0 = [br.read(IB if i != 0 else IB - 1) for i in range(16)]
        idx1 = [br.read(IB2 if i != 0 else IB2 - 1) for i in range(16)]
        if isb == 0:
            color_idx, color_bits, alpha_idx, alpha_bits = idx0, IB, idx1, IB2
        else:
            color_idx, color_bits, alpha_idx, alpha_bits = idx1, IB2, idx0, IB
        for i in range(16):
            rgb = interpolate(e0[:3], e1[:3], WEIGHTS[color_bits][color_idx[i]])
            a = interpolate(e0[3:4], e1[3:4], WEIGHTS[alpha_bits][alpha_idx[i]])[0]
            pixels[i] = (*rgb, a)

    if rot:
        pixels = [apply_rotation(p, rot) for p in pixels]
    return pixels


def decode_block_approx_multisubset(br, params):
    br.read(params["PB"])
    br.read(params["RB"])
    if params["ISB"]:
        br.read(1)
    NS = params["NS"]
    CB, AB = params["CB"], params["AB"]
    sums = [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    for ci, bits in enumerate((CB, CB, CB, AB if AB else 0)):
        if bits == 0:
            continue
        for _s in range(NS):
            for _e in range(2):
                sums[ci] += expand_bits(br.read(bits), bits)
                counts[ci] += 1
    r = sums[0] // counts[0] if counts[0] else 0
    g = sums[1] // counts[1] if counts[1] else 0
    b = sums[2] // counts[2] if counts[2] else 0
    a = sums[3] // counts[3] if counts[3] else 255
    return [(r, g, b, a)] * 16


def decode_bc7_block(block16):
    br = BitReader(block16)
    mode = -1
    for i in range(8):
        if (br.v >> i) & 1:
            mode = i
            break
    if mode == -1:
        return [(0, 0, 0, 255)] * 16
    br.pos = mode + 1
    params = MODE_PARAMS[mode]
    if params["NS"] == 1:
        return decode_block_exact_ns1(br, params)
    return decode_block_approx_multisubset(br, params)


def decode_bc7_image_py(image_data, width, height):
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    out = Image.new("RGBA", (blocks_x * 4, blocks_y * 4))
    px = out.load()
    pos = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            pixels = decode_bc7_block(image_data[pos:pos + 16])
            pos += 16
            for i, p in enumerate(pixels):
                px[bx * 4 + (i % 4), by * 4 + (i // 4)] = p
    return out.crop((0, 0, width, height))


# ---------- запасной питоновский декодер BC4 ----------

def decode_bc4_block(block8):
    r0, r1 = block8[0], block8[1]
    if r0 > r1:
        ramp = [r0, r1,
                (6 * r0 + 1 * r1) // 7, (5 * r0 + 2 * r1) // 7,
                (4 * r0 + 3 * r1) // 7, (3 * r0 + 4 * r1) // 7,
                (2 * r0 + 5 * r1) // 7, (1 * r0 + 6 * r1) // 7]
    else:
        ramp = [r0, r1,
                (4 * r0 + 1 * r1) // 5, (3 * r0 + 2 * r1) // 5,
                (2 * r0 + 3 * r1) // 5, (1 * r0 + 4 * r1) // 5,
                0, 255]
    bits = int.from_bytes(block8[2:8], "little")
    return [ramp[(bits >> (3 * i)) & 0x7] for i in range(16)]


def decode_bc4_image_py(image_data, width, height):
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    out = Image.new("L", (blocks_x * 4, blocks_y * 4))
    px = out.load()
    pos = 0
    for by in range(blocks_y):
        for bx in range(blocks_x):
            values = decode_bc4_block(image_data[pos:pos + 8])
            pos += 8
            for i, v in enumerate(values):
                px[bx * 4 + (i % 4), by * 4 + (i // 4)] = v
    return out.crop((0, 0, width, height)).convert("RGBA")


def decode_raw_rgba8_image(image_data, width, height):
    expected = width * height * 4
    data = bytes(image_data[:expected])
    if len(data) < expected:
        data += b"\x00" * (expected - len(data))
    return Image.frombytes("RGBA", (width, height), data)


def decode_image(kind, image_data, width, height):
    """Декодирует mip0 в RGBA-картинку."""
    if kind == "rgba8":
        return decode_raw_rgba8_image(image_data, width, height)
    n = 7 if kind == "bc7" else 4
    try:
        return _pil_bcn_decode(image_data, width, height, n)
    except Exception:
        # очень старый Pillow без нужного декодера - считаем по-старому
        if kind == "bc7":
            return decode_bc7_image_py(image_data, width, height)
        return decode_bc4_image_py(image_data, width, height)


def decode_layers(kind, image_data, width, height, layers):
    """Декодирует mip0 со всеми слоями. Слои кладутся рядом слева направо,
    так что картинка получается шириной width * layers."""
    if layers <= 1:
        return decode_image(kind, image_data, width, height)
    per = len(image_data) // layers
    atlas = Image.new("RGBA", (width * layers, height))
    for i in range(layers):
        layer = decode_image(kind, image_data[i * per:(i + 1) * per], width, height)
        atlas.paste(layer, (i * width, 0))
    return atlas


def split_layers(img, layers):
    """Режет ленту обратно на слои одинаковой ширины."""
    if layers <= 1:
        return [img]
    if img.width % layers:
        raise ValueError(f"В оригинале {layers} слоя(ёв): ширина картинки ({img.width}) "
                         f"должна делиться на {layers} - слои лежат рядом слева направо.")
    w = img.width // layers
    return [img.crop((i * w, 0, (i + 1) * w, img.height)) for i in range(layers)]


def resize_layers(img, layers, layer_size, resample):
    """Масштабирует каждый слой отдельно, чтобы соседние слои не смешивались на стыке."""
    if layers <= 1:
        return resize_image(img, layer_size, resample)
    out = Image.new(img.mode, (layer_size[0] * layers, layer_size[1]))
    for i, layer in enumerate(split_layers(img, layers)):
        out.paste(resize_image(layer, layer_size, resample), (i * layer_size[0], 0))
    return out


# ============================================================
#  Энкодеры (NumPy, по полосам, параллельно)
# ============================================================

_CHUNK_BLOCKS = 16384          # сколько блоков считаем за один заход NumPy
_STRIP_BLOCKS = 32768          # примерный размер одной задачи для пула
_PARALLEL_MIN_BLOCKS = 65536   # меньше этого - пул не нужен, быстрее посчитать на месте

_ALPHA_W = np.array([1, 1, 1, 4], dtype=np.int32)


def _to_blocks(arr):
    """(H, W, C) или (H, W) c H, W кратными 4 -> (N, 16, C) / (N, 16)."""
    h, w = arr.shape[:2]
    if arr.ndim == 3:
        c = arr.shape[2]
        return arr.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)
    return arr.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)


def _quant_endpoint(v):
    """7 бит на канал + общий p-бит. v: (N, 4) int32 -> (raw (N, 4), pbit (N,))."""
    best_raw = best_err = best_pb = None
    for pb in (0, 1):
        raw = np.clip((v - pb + 1) // 2, 0, 127)
        # ошибка по альфе весит больше: 255 должно остаться 255, а 0 - нулём
        err = (np.abs(raw * 2 + pb - v) * _ALPHA_W).sum(axis=1)
        if best_raw is None:
            best_raw, best_err = raw, err
            best_pb = np.zeros(len(v), dtype=np.int32)
        else:
            better = err < best_err
            best_raw = np.where(better[:, None], raw, best_raw)
            best_pb = better.astype(np.int32)
    return best_raw, best_pb


def _bc7_mode6_blocks(blocks):
    """blocks: (N, 16, 4) uint8 -> N * 16 байт BC7 (mode 6)."""
    n = len(blocks)
    p = blocks.astype(np.float32)
    lo = p.min(axis=1)
    hi = p.max(axis=1)

    # Концы отрезка - углы bounding box. Какой именно угол брать по каждому
    # каналу, решает знак ковариации с "главным" каналом (иначе каналы,
    # которые меняются в противофазе, превращаются в кашу).
    d = p - p.mean(axis=1, keepdims=True)
    main = (hi - lo).argmax(axis=1)
    dm = np.take_along_axis(d, main[:, None, None], axis=2)
    flip = (d * dm).sum(axis=1) < 0
    e0 = np.where(flip, hi, lo).astype(np.int32)
    e1 = np.where(flip, lo, hi).astype(np.int32)

    raw0, pb0 = _quant_endpoint(e0)
    raw1, pb1 = _quant_endpoint(e1)
    q0 = (raw0 * 2 + pb0[:, None]).astype(np.float32)
    q1 = (raw1 * 2 + pb1[:, None]).astype(np.float32)

    # Индекс пикселя = проекция на отрезок (веса BC7 - это round(64*i/15))
    dirv = q1 - q0
    len2 = np.maximum((dirv * dirv).sum(axis=1), 1.0)
    t = ((p - q0[:, None, :]) * dirv[:, None, :]).sum(axis=2) / len2[:, None]
    idx = np.clip(np.rint(t * 15.0), 0, 15).astype(np.int64)

    # У первого пикселя только 3 бита: если его индекс >= 8, меняем концы местами
    swap = idx[:, 0] >= 8
    idx = np.where(swap[:, None], 15 - idx, idx).astype(np.uint64)
    sw = swap[:, None]
    raw0, raw1 = np.where(sw, raw1, raw0), np.where(sw, raw0, raw1)
    pb0, pb1 = np.where(swap, pb1, pb0), np.where(swap, pb0, pb1)
    raw0 = raw0.astype(np.uint64)
    raw1 = raw1.astype(np.uint64)
    pb0 = pb0.astype(np.uint64)
    pb1 = pb1.astype(np.uint64)

    lo64 = np.full(n, 0x40, dtype=np.uint64)           # бит 6 = mode 6
    for c in range(4):
        lo64 |= raw0[:, c] << np.uint64(7 + 14 * c)
        lo64 |= raw1[:, c] << np.uint64(14 + 14 * c)
    lo64 |= pb0 << np.uint64(63)
    hi64 = pb1 | (idx[:, 0] << np.uint64(1))
    for i in range(1, 16):
        hi64 |= idx[:, i] << np.uint64(4 * i)

    out = np.empty((n, 2), dtype="<u8")
    out[:, 0] = lo64
    out[:, 1] = hi64
    return out.tobytes()


def _bc4_blocks(blocks):
    """blocks: (N, 16) uint8 -> N * 8 байт BC4."""
    v = blocks.astype(np.float32)
    r0 = v.max(axis=1)
    r1 = v.min(axis=1)
    span = np.maximum(r0 - r1, 1.0)
    k = np.clip(np.rint((r0[:, None] - v) / span[:, None] * 7.0), 0, 7).astype(np.int64)
    # k=0 -> r0 (индекс 0), k=7 -> r1 (индекс 1), между ними - индексы 2..7
    idx = np.where(k == 0, 0, np.where(k == 7, 1, k + 1))
    idx = np.where((r0 == r1)[:, None], 0, idx).astype(np.uint64)
    word = r0.astype(np.uint64) | (r1.astype(np.uint64) << np.uint64(8))
    for i in range(16):
        word |= idx[:, i] << np.uint64(16 + 3 * i)
    return word.astype("<u8").tobytes()


def _encode_strip(kind, arr):
    """Задача для рабочего процесса: кодирует полосу картинки."""
    blocks = _to_blocks(arr)
    fn = _bc7_mode6_blocks if kind == "bc7" else _bc4_blocks
    return b"".join(fn(blocks[i:i + _CHUNK_BLOCKS]) for i in range(0, len(blocks), _CHUNK_BLOCKS))


# ---------- пул процессов ----------

_pool = None
_pool_lock = threading.Lock()


def worker_count():
    return max(1, min(os.cpu_count() or 1, 61))   # 61 - предел Windows


def get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=worker_count())
        return _pool


def shutdown_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None


def _run_strips(executor, kind, arr, strips, on_blocks):
    bw = arr.shape[1] // 4
    futures = {executor.submit(_encode_strip, kind, arr[y0:y1]): i
               for i, (y0, y1) in enumerate(strips)}
    parts = [None] * len(strips)
    for fut in as_completed(futures):
        i = futures[fut]
        parts[i] = fut.result()
        on_blocks((strips[i][1] - strips[i][0]) // 4 * bw)
    return b"".join(parts)


def encode_level(img, kind, on_blocks=lambda n: None, parallel=True):
    """Кодирует один мип-уровень. on_blocks(n) вызывается по мере готовности блоков.

    parallel=False - не использовать пул процессов (нужно, когда файлы уже
    параллелятся на уровне массовой обработки, иначе процессы множатся)."""
    if kind == "rgba8":
        data = img.convert("RGBA").tobytes()
        on_blocks(((img.width + 3) // 4) * ((img.height + 3) // 4))
        return data

    arr = np.asarray(img.convert("RGBA" if kind == "bc7" else "L"))
    h, w = arr.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        pad = ((0, ph), (0, pw)) + (((0, 0),) if arr.ndim == 3 else ())
        arr = np.pad(arr, pad, mode="edge")
    bh, bw = arr.shape[0] // 4, arr.shape[1] // 4

    if not parallel or bh * bw < _PARALLEL_MIN_BLOCKS or worker_count() == 1:
        data = _encode_strip(kind, arr)
        on_blocks(bh * bw)
        return data

    rows = max(1, _STRIP_BLOCKS // bw)
    strips = [(y * 4, min(bh, y + rows) * 4) for y in range(0, bh, rows)]
    counted = [0]

    def counting(nblocks):
        counted[0] += nblocks
        on_blocks(nblocks)

    try:
        return _run_strips(get_pool(), kind, arr, strips, counting)
    except Exception:
        # Процессы не запустились (редко, но бывает) - считаем потоками:
        # NumPy отпускает GIL, так что ускорение всё равно есть.
        shutdown_pool()
        on_blocks(-counted[0])
        with ThreadPoolExecutor(max_workers=worker_count()) as tp:
            return _run_strips(tp, kind, arr, strips, on_blocks)


# ============================================================
#  Масштабирование и сборка KTX
# ============================================================

def resize_image(img, size, resample):
    """Масштабирует каждый канал отдельно и параллельно.

    Отдельно - чтобы цвет под прозрачными пикселями не обнулялся
    (обычный RGBA-resize в Pillow домножает цвет на альфу)."""
    if img.size == tuple(size):
        return img
    bands = img.split()
    if len(bands) == 1:
        return img.resize(size, resample)
    with ThreadPoolExecutor(max_workers=len(bands)) as tp:
        resized = list(tp.map(lambda b: b.resize(size, resample), bands))
    return Image.merge(img.mode, resized)


def target_size(width, height, long_side):
    """Размер после растяжения: длинная сторона = long_side, пропорции сохраняются."""
    if not long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def mip_level_count(orig_header, new_w, new_h):
    orig_levels = orig_header['numberOfMipmapLevels']
    if orig_levels <= 1:
        return 1
    full = int(math.floor(math.log2(max(new_w, new_h)))) + 1
    orig_max = max(orig_header['pixelWidth'], orig_header['pixelHeight'], 1)
    shift = int(round(math.log2(max(new_w, new_h) / orig_max)))
    return max(1, min(full, orig_levels + shift))


def build_ktx(image, original_ktx_path, progress=lambda frac, text: None, parallel=True):
    """Собирает новый .ktx с заголовком оригинала и картинкой image.

    Возвращает dict: bytes, mip0 (сжатые данные верхнего уровня), kind,
    fmt_name, width, height, levels. parallel=False - кодировать в текущем
    процессе, без собственного пула (для массовой обработки)."""
    with open(original_ktx_path, "rb") as f:
        orig = f.read()
    header_d, _, header_len = parse_ktx_bytes(orig)
    check_simple_2d(header_d)
    fmt_name, kind = FORMATS[header_d['glInternalFormat']]

    layers = layer_count(header_d)
    layer_imgs = split_layers(image.convert("L" if kind == "bc4" else "RGBA"), layers)
    w, h = layer_imgs[0].size
    levels = mip_level_count(header_d, w, h)

    sizes = [(max(1, w >> i), max(1, h >> i)) for i in range(levels)]
    total_blocks = layers * sum(((mw + 3) // 4) * ((mh + 3) // 4) for mw, mh in sizes)
    done = [0]

    def on_blocks(nblocks):
        done[0] += nblocks
        progress(done[0] / total_blocks, f"Кодирование {fmt_name}: {w}x{h}")

    header = bytearray(orig[:header_len])
    struct.pack_into("<I", header, 16 + 20, w)
    struct.pack_into("<I", header, 16 + 24, h)
    if header_d['numberOfMipmapLevels'] > 1:
        struct.pack_into("<I", header, 16 + 40, levels)

    out = [bytes(header)]
    mip0 = None
    for i, (mw, mh) in enumerate(sizes):
        if i > 0:
            layer_imgs = [resize_image(li, (mw, mh), _RES.BOX) for li in layer_imgs]
        # в KTX слои одного мип-уровня идут подряд под общим imageSize
        data = b"".join(encode_level(li, kind, on_blocks, parallel) for li in layer_imgs)
        if i == 0:
            mip0 = data
        out.append(struct.pack("<I", len(data)))
        out.append(data)
        if len(data) % 4:
            out.append(b"\x00" * (4 - len(data) % 4))

    return dict(bytes=b"".join(out), mip0=mip0, kind=kind, fmt_name=fmt_name,
                width=w, height=h, levels=levels, layers=layers)


def repack_ktx_bytes(new_image, original_ktx_path):
    """Совместимость со старым именем функции."""
    return build_ktx(new_image, original_ktx_path)["bytes"]


# ============================================================
#  Массовая обработка папок игры
#  (файлы параллелятся между процессами: один файл = один процесс,
#   внутри процесса кодирование идёт без собственного пула)
# ============================================================

SLOTS = ("A", "B")                 # две папки текстур игры
BACKUP_DIRNAME = "ktx_backups"


def app_dir():
    """Папка рядом со скриптом (или с .exe, если собрано через PyInstaller)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def settings_path():
    return os.path.join(app_dir(), "ktx_tool_settings.json")


def fmt_time(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_size(n):
    n = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024


def free_bytes(path):
    """Свободное место на диске, где лежит (или будет лежать) path."""
    p = os.path.abspath(path) if path else app_dir()
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return shutil.disk_usage(p).free
    except Exception:
        return None


def available_ram():
    """Свободная оперативная память в байтах (None, если не удалось узнать)."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MEMSTATUS()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullAvailPhys)
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def auto_workers(max_pixels):
    """Сколько процессов можно запустить: меньше из числа ядер и того, что влезает в RAM.

    В пике один процесс держит около 4-5 копий картинки RGBA (исходник, растянутая,
    копия под кодирование, массив блоков) - для 8K это больше гигабайта."""
    cpu = worker_count()
    ram = available_ram()
    if not ram:
        return min(cpu, 4)
    per_worker = max(max_pixels * 4 * 5, 128 * 1024 * 1024)
    return max(1, min(cpu, int(ram * 0.7 // per_worker)))


def folder_tag(slot, root):
    """Имя подпапки для бэкапов/вывода: 'A_<имя папки>' (A и B с одинаковыми именами файлов не смешиваются)."""
    name = os.path.basename(os.path.normpath(root)) or "root"
    return f"{slot}_{name}"


def est_ktx_bytes(w, h, layers, kind, mips):
    bpp = {"bc7": 1.0, "bc4": 0.5, "rgba8": 4.0}[kind]
    return int(w * h * layers * bpp * (4 / 3 if mips > 1 else 1))


def _scan_one(slot, tag, root, full, rel):
    it = dict(slot=slot, tag=tag, root=root, path=full, rel=rel, ok=False, err="",
              w=0, h=0, layers=1, fmt="", kind="", mips=1, size=0)
    try:
        it["size"] = os.path.getsize(full)
        with open(full, "rb") as f:
            head = f.read(64)
        if len(head) < 64 or head[:12] != KTX_IDENTIFIER:
            raise ValueError("не KTX v1")
        if struct.unpack_from("<I", head, 12)[0] != 0x04030201:
            raise ValueError("big-endian не поддерживается")
        hd = dict(zip(HEADER_NAMES, struct.unpack_from("<12I", head, 16)))
        if hd['glInternalFormat'] not in FORMATS:
            raise ValueError(f"формат 0x{hd['glInternalFormat']:X} не поддерживается")
        check_simple_2d(hd)
        if hd['pixelWidth'] < 1 or hd['pixelHeight'] < 1:
            raise ValueError("нулевой размер (1D-текстура?)")
        fmt_name, kind = FORMATS[hd['glInternalFormat']]
        it.update(ok=True, w=hd['pixelWidth'], h=hd['pixelHeight'], layers=layer_count(hd),
                  fmt=fmt_name, kind=kind, mips=max(1, hd['numberOfMipmapLevels']))
    except Exception as e:  # noqa: BLE001
        it["err"] = str(e).replace("\n", " ")
    return it


def scan_ktx_folders(folders, exclude=()):
    """folders: [(slot, путь)]. Рекурсивно ищет .ktx и читает их заголовки (быстро)."""
    excl = {os.path.normcase(os.path.abspath(p)) for p in exclude if p}
    items = []
    for slot, root in folders:
        tag = folder_tag(slot, root)
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs
                       if os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) not in excl]
            for name in files:
                if name.lower().endswith(".ktx"):
                    full = os.path.join(dirpath, name)
                    items.append(_scan_one(slot, tag, root, full, os.path.relpath(full, root)))
    items.sort(key=lambda it: (it["slot"], it["rel"].lower()))
    return items


# ---------- запись, бэкапы ----------

def _atomic_write(path, data):
    """Пишем во временный файл и подменяем - обрыв посередине не оставит битую текстуру."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp~"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _backup_once(src, backup_path):
    """Копирует оригинал в бэкап ТОЛЬКО если там ещё ничего нет: повторный запуск
    не должен затереть чистый оригинал уже изменённым файлом."""
    if os.path.exists(backup_path):
        return
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    tmp = backup_path + ".tmp~"
    shutil.copy2(src, tmp)
    os.replace(tmp, backup_path)


def _write_result(job, data):
    src, dst = job["path"], job["dst"]
    same = os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst))
    if job.get("backup") and same:
        _backup_once(src, job["backup"])
    _atomic_write(dst, data)


# ---------- рисование номера для режима изучения ----------

@functools.lru_cache(maxsize=64)
def _number_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "LiberationSans-Bold.ttf", "Verdana.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ImageFont.load_default(size)      # Pillow >= 10.1: масштабируемый шрифт
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()          # совсем старый Pillow: мелкий запасной шрифт


def number_color(number, kind):
    """Цвет фона для номера: соседние номера заметно различаются по оттенку."""
    if kind == "bc4":                            # BC4 - один канал, оттенок не виден
        g = 40 + (number * 53) % 110
        return (g, g, g)
    r, g, b = colorsys.hsv_to_rgb((number * 0.61803398875) % 1.0, 0.65, 0.55)
    return (int(r * 255), int(g * 255), int(b * 255))


def _text_box(draw, text, font, stroke):
    try:
        return draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    except Exception:  # noqa: BLE001
        return draw.textbbox((0, 0), text, font=font)


def _draw_text(draw, xy, text, font, stroke):
    try:
        draw.text(xy, text, font=font, fill=(255, 255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    except Exception:  # noqa: BLE001
        draw.text(xy, text, font=font, fill=(255, 255, 255, 255))


def render_number_image(w, h, text, color):
    """Картинка w x h: цветной фон, рамка, большой номер по центру, маленькие номера
    в углах (виден даже если текстура растянута/обрезана) и жёлтый треугольник сверху
    (видно, не перевёрнута ли текстура)."""
    img = Image.new("RGBA", (w, h), tuple(color) + (255,))
    m = min(w, h)
    if m < 12:
        return img                               # крошечная текстура: различаем только по цвету
    d = ImageDraw.Draw(img)
    bw = max(1, m // 64)
    d.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255, 255), width=bw)

    size = max(8, int(m * 0.5))
    while True:
        font = _number_font(size)
        stroke = max(1, size // 14)
        bb = _text_box(d, text, font, stroke)
        if (bb[2] - bb[0] <= w * 0.85 and bb[3] - bb[1] <= h * 0.85) or size <= 8:
            break
        size = max(8, int(size * 0.85))
    x = (w - (bb[2] - bb[0])) // 2 - bb[0]
    y = (h - (bb[3] - bb[1])) // 2 - bb[1]
    _draw_text(d, (x, y), text, font, stroke)

    if m >= 64:
        small = max(8, m // 9)
        sfont = _number_font(small)
        sstroke = max(1, small // 12)
        sb = _text_box(d, text, sfont, sstroke)
        sw, sh = sb[2] - sb[0], sb[3] - sb[1]
        pad = bw + 3
        for cx, cy in ((pad, pad), (w - sw - pad, pad), (pad, h - sh - pad), (w - sw - pad, h - sh - pad)):
            _draw_text(d, (cx - sb[0], cy - sb[1]), text, sfont, sstroke)
        tri = max(6, m // 12)
        top = bw + 3
        d.polygon([(w // 2, top), (w // 2 - tri // 2, top + tri), (w // 2 + tri // 2, top + tri)],
                  fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))
    return img


# ---------- воркеры (выполняются в отдельных процессах) ----------

def _guard(fn):
    """Любая ошибка файла превращается в строку результата, а не роняет весь пул."""
    @functools.wraps(fn)
    def inner(job):
        t0 = time.time()
        try:
            res = fn(job)
        except MemoryError:
            res = dict(status="err", msg="не хватило памяти - уменьши число потоков")
        except Exception as e:  # noqa: BLE001
            res = dict(status="err", msg=f"{type(e).__name__}: {e}".replace("\n", " "))
        res["sec"] = round(time.time() - t0, 2)
        return res
    return inner


@_guard
def _w_decode(job):
    header, image_data = parse_ktx(job["path"])
    check_simple_2d(header)
    w, h = header['pixelWidth'], header['pixelHeight']
    fmt_name, kind = FORMATS[header['glInternalFormat']]
    layers = layer_count(header)
    img = decode_layers(kind, image_data, w, h, layers)
    os.makedirs(os.path.dirname(job["dst"]) or ".", exist_ok=True)
    img.save(job["dst"], compress_level=1)       # 1 - быстро; для 8K это секунды против десятков секунд
    return dict(status="ok", msg=f"{w}x{h}  {fmt_name}" + (f"  слоёв {layers}" if layers > 1 else ""))


@_guard
def _w_rescale(job):
    header, image_data = parse_ktx(job["path"])
    check_simple_2d(header)
    w, h = header['pixelWidth'], header['pixelHeight']
    fmt_name, kind = FORMATS[header['glInternalFormat']]
    layers = layer_count(header)
    img = decode_layers(kind, image_data, w, h, layers)
    del image_data
    img = resize_layers(img, layers, (job["tw"], job["th"]), FILTER_CHOICES[job["filter"]])
    res = build_ktx(img, job["path"], parallel=False)
    del img
    _write_result(job, res["bytes"])
    return dict(status="ok",
                msg=f"{w}x{h} -> {res['width']}x{res['height']}  {fmt_name}  {fmt_size(len(res['bytes']))}")


_SINGLE_CACHE = {}


def _single_source(path, size, filt):
    """Общая картинка, растянутая до size. Кэш на процесс: размеры текстур часто повторяются."""
    key = (path, os.path.getmtime(path))
    if _SINGLE_CACHE.get("key") != key:
        _SINGLE_CACHE.clear()
        img = Image.open(path)
        img.load()
        _SINGLE_CACHE.update(key=key, src=img.convert("RGBA"), sized={})
    src, sized = _SINGLE_CACHE["src"], _SINGLE_CACHE["sized"]
    rk = (tuple(size), filt)
    if rk in sized:
        return sized[rk]
    out = resize_image(src, size, FILTER_CHOICES[filt])
    if size[0] * size[1] <= 1024 * 1024:         # крупные не кэшируем - это сотни МБ на процесс
        if len(sized) >= 16:
            sized.clear()
        sized[rk] = out
    return out


@_guard
def _w_single(job):
    tw, th, layers = job["tw"], job["th"], job["layers"]
    layer = _single_source(job["image"], (tw, th), job["filter"])
    if layers == 1:
        atlas = layer
    else:
        atlas = Image.new("RGBA", (tw * layers, th))
        for i in range(layers):
            atlas.paste(layer, (i * tw, 0))
    res = build_ktx(atlas, job["path"], parallel=False)
    _write_result(job, res["bytes"])
    return dict(status="ok", msg=f"{tw}x{th}  {job['fmt']}  {fmt_size(len(res['bytes']))}")


@_guard
def _w_number(job):
    w, h, layers, n = job["tw"], job["th"], job["layers"], job["number"]
    color = number_color(n, job["kind"])
    if layers == 1:
        atlas = render_number_image(w, h, str(n), color)
    else:
        atlas = Image.new("RGBA", (w * layers, h))
        for i in range(layers):
            atlas.paste(render_number_image(w, h, f"{n}.{i + 1}", color), (i * w, 0))
    res = build_ktx(atlas, job["path"], parallel=False)
    _write_result(job, res["bytes"])
    return dict(status="ok", msg=f"#{n}  {w}x{h}  {job['fmt']}")


WORKERS = {"rescale": _w_rescale, "single": _w_single, "number": _w_number}


# ---------- планирование и запуск ----------

def plan_encode_jobs(items, mode, o):
    """Превращает результат сканирования в список заданий.

    o: long_side, skip_ge, min_src, out_dir, backup_dir, filter, image."""
    jobs, skipped, bad = [], [], []
    est_new = est_old = 0
    number = 0
    for it in items:
        if not it["ok"]:
            bad.append(it)
            continue
        w, h = it["w"], it["h"]
        if mode == "rescale":
            if max(w, h) < o["min_src"]:
                skipped.append((it, f"сторона меньше {o['min_src']}"))
                continue
            if o["skip_ge"] and max(w, h) >= o["long_side"]:
                skipped.append((it, "уже такого размера или больше"))
                continue
            tw, th = target_size(w, h, o["long_side"])
        elif mode == "single":
            tw, th = target_size(w, h, o["long_side"]) if o["long_side"] else (w, h)
        else:
            tw, th = w, h
        job = dict(it)
        job.update(mode=mode, tw=tw, th=th, cost=tw * th * it["layers"], filter=o["filter"],
                   image=o["image"],
                   dst=os.path.join(o["out_dir"], it["tag"], it["rel"]) if o["out_dir"] else it["path"],
                   backup=(os.path.join(o["backup_dir"], it["tag"], it["rel"])
                           if o["backup_dir"] and not o["out_dir"] else None))
        if mode == "number":
            number += 1
            job["number"] = number
        jobs.append(job)
        est_new += est_ktx_bytes(tw, th, it["layers"], it["kind"], it["mips"])
        est_old += it["size"]
    return dict(jobs=jobs, skipped=skipped, bad=bad, est_new=est_new, est_old=est_old)


def run_batch(worker, jobs, workers, progress, log, cancel, label):
    """Гоняет worker(job) по всем заданиям в пуле процессов.

    cancel - threading.Event: после его установки новые файлы не стартуют, уже
    идущие дописываются. Возвращает dict(ok, skip, err, total, results, elapsed, cancelled)."""
    n = len(jobs)
    total_cost = sum(j["cost"] for j in jobs) or 1
    st = dict(done=0, cost=0, ok=0, skip=0, err=0)
    results = []
    t0 = time.time()
    jobs = sorted(jobs, key=lambda j: -j["cost"])        # крупные первыми - ровнее загрузка ядер

    def finish(job, res):
        st["done"] += 1
        st["cost"] += job["cost"]
        st[res["status"]] += 1
        res.update(tag=job["tag"], rel=job["rel"], number=job.get("number"))
        results.append(res)
        mark = {"ok": "OK  ", "skip": "SKIP", "err": "ERR "}[res["status"]]
        log(f"[{mark}] {job['tag']}/{job['rel']}  {res.get('msg', '')}")
        el = time.time() - t0
        frac = st["cost"] / total_cost
        eta = f"  |  осталось ~{fmt_time(el / frac * (1 - frac))}" if frac > 0.02 else ""
        progress(frac, f"{label}: {st['done']}/{n}  |  {fmt_time(el)}{eta}")

    cancelled = False
    if cancel.is_set():                                    # отменили ещё до старта
        return dict(ok=0, skip=0, err=0, total=n, results=[], elapsed=0.0, cancelled=True)
    if workers <= 1:
        for job in jobs:
            if cancel.is_set():
                cancelled = True
                break
            finish(job, worker(job))
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        futs = {ex.submit(worker, job): i for i, job in enumerate(jobs)}
        seen = set()
        try:
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    log("!!! Рабочий процесс аварийно завершился (чаще всего не хватило памяти). "
                        "Уменьши число потоков или целевое разрешение и запусти ещё раз.")
                    cancelled = True
                    break
                except Exception as e:  # noqa: BLE001
                    res = dict(status="err", msg=f"{type(e).__name__}: {e}")
                seen.add(i)
                finish(jobs[i], res)
                if cancel.is_set():
                    cancelled = True
                    break
        finally:
            for f in futs:
                f.cancel()
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except TypeError:                              # Python 3.8
                ex.shutdown(wait=True)
        if cancelled:                                      # те, что успели дописаться при остановке
            for fut, i in futs.items():
                if i not in seen and fut.done() and not fut.cancelled():
                    try:
                        finish(jobs[i], fut.result())
                    except Exception:  # noqa: BLE001
                        pass
    return dict(ok=st["ok"], skip=st["skip"], err=st["err"], total=n, results=results,
                elapsed=time.time() - t0, cancelled=cancelled)


def write_numbering_log(path, jobs, results):
    """CSV (разделитель ;, открывается в Excel): какой номер какой текстуре достался."""
    by_key = {(r["tag"], r["rel"]): r for r in results}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["номер", "папка", "файл", "размер", "формат", "слоёв", "цвет фона", "статус"])
        for j in sorted(jobs, key=lambda j: j["number"]):
            r = by_key.get((j["tag"], j["rel"]))
            status = "не обработан" if r is None else ("ok" if r["status"] == "ok" else f"{r['status']}: {r.get('msg', '')}")
            wr.writerow([j["number"], j["tag"], j["rel"], f"{j['w']}x{j['h']}", j["fmt"], j["layers"],
                         "#%02x%02x%02x" % number_color(j["number"], j["kind"]), status])


def plan_restore(folders, backup_dir):
    """Пары (файл в бэкапе, куда вернуть) для всех .ktx из папки бэкапов."""
    pairs = []
    for slot, root in folders:
        bdir = os.path.join(backup_dir, folder_tag(slot, root))
        if not os.path.isdir(bdir):
            continue
        for dirpath, _dirs, files in os.walk(bdir):
            for name in files:
                if name.lower().endswith(".ktx"):
                    bpath = os.path.join(dirpath, name)
                    pairs.append((bpath, os.path.join(root, os.path.relpath(bpath, bdir))))
    return pairs


def restore_files(pairs, progress, log, cancel):
    ok = err = 0
    for i, (src, dst) in enumerate(pairs):
        if cancel.is_set():
            break
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            tmp = dst + ".tmp~"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            ok += 1
        except Exception as e:  # noqa: BLE001
            err += 1
            log(f"[ERR ] {dst}: {e}")
        progress((i + 1) / len(pairs), f"Восстановление: {i + 1}/{len(pairs)}")
    return ok, err


# ============================================================
#  GUI
# ============================================================

class KtxToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KTX Texture Tool - для личного использования")
        self.geometry("980x860")      # массовые вкладки выше обычных
        self.resizable(True, True)

        self._busy = False

        self.decoded_image = None
        self.decode_source_path = None

        self.new_image = None
        self.new_image_path = None
        self.encode_original_ktx_path = None
        self._encoded_bytes = None

        self.upscale_source_path = None
        self._upscaled_bytes = None

        # --- массовые вкладки: общие настройки (запоминаются между запусками) ---
        self._cancel = threading.Event()
        self._log_q = queue.Queue()
        self.var_dirA = tk.StringVar()
        self.var_dirB = tk.StringVar()
        self.var_decout = tk.StringVar()
        self.var_encout = tk.StringVar()
        self.var_target = tk.StringVar(value="game")         # game | folder
        self.var_do_backup = tk.BooleanVar(value=True)
        self.var_backup = tk.StringVar(value=os.path.join(app_dir(), BACKUP_DIRNAME))
        self.var_mode = tk.StringVar(value="rescale")        # rescale | single | number
        self.var_bsize = tk.StringVar(value="8K (8192)")
        self.var_bfilter = tk.StringVar(value="Lanczos (плавно)")
        self.var_skip_ge = tk.BooleanVar(value=True)
        self.var_minsrc = tk.StringVar(value="0")
        self.var_single_img = tk.StringVar()
        self.var_logdir = tk.StringVar(value=app_dir())
        self.var_workers = tk.StringVar(value="0")
        self._settings_vars = {
            "dirA": self.var_dirA, "dirB": self.var_dirB, "decout": self.var_decout,
            "encout": self.var_encout, "target": self.var_target, "do_backup": self.var_do_backup,
            "backup": self.var_backup, "mode": self.var_mode, "bsize": self.var_bsize,
            "bfilter": self.var_bfilter, "skip_ge": self.var_skip_ge, "minsrc": self.var_minsrc,
            "single_img": self.var_single_img, "logdir": self.var_logdir, "workers": self.var_workers,
        }
        self._load_settings()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        self.decode_tab = ttk.Frame(notebook)
        self.encode_tab = ttk.Frame(notebook)
        self.upscale_tab = ttk.Frame(notebook)
        self.bdec_tab = ttk.Frame(notebook)
        self.benc_tab = ttk.Frame(notebook)
        notebook.add(self.decode_tab, text="Decode (.ktx -> PNG)")
        notebook.add(self.encode_tab, text="Encode (PNG -> .ktx)")
        notebook.add(self.upscale_tab, text="Upscale 8K (.ktx -> .ktx)")
        notebook.add(self.bdec_tab, text="Массовый Decode")
        notebook.add(self.benc_tab, text="Массовый Encode")

        self._build_decode_tab()
        self._build_encode_tab()
        self._build_upscale_tab()
        self._build_bdec_tab()
        self._build_benc_tab()
        self.after(100, self._poll_log)

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(status, maximum=1000, length=260)
        self.progress.pack(side="left")
        self.status_label = ttk.Label(status, text=f"Готов. Ядер для кодирования: {worker_count()}")
        self.status_label.pack(side="left", padx=10)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._cancel.set()
        self._save_settings()
        shutdown_pool()
        self.destroy()

    # ---------------- фоновые задачи ----------------
    def _run_bg(self, work, on_done, error_title="Ошибка"):
        """work(progress) выполняется в фоновом потоке, on_done(result) - в GUI."""
        if self._busy:
            messagebox.showinfo("Подожди", "Предыдущая операция ещё выполняется.")
            return
        self._busy = True
        q = queue.Queue()
        started = time.time()

        def progress(frac, text):
            q.put(("p", frac, text))

        def runner():
            try:
                q.put(("ok", work(progress)))
            except Exception as e:  # noqa: BLE001 - показываем пользователю как есть
                q.put(("err", e))

        def poll():
            last = None
            finished = None
            try:
                while True:
                    item = q.get_nowait()
                    if item[0] == "p":
                        last = item
                    else:
                        finished = item
                        break
            except queue.Empty:
                pass
            if last is not None:
                self.progress["value"] = int(max(0.0, min(1.0, last[1])) * 1000)
                self.status_label.config(text=last[2])
            if finished is None:
                self.after(40, poll)
                return
            self._busy = False
            elapsed = time.time() - started
            if finished[0] == "ok":
                self.progress["value"] = 1000
                self.status_label.config(text=f"Готово за {elapsed:.1f} с")
                on_done(finished[1])
            else:
                self.progress["value"] = 0
                self.status_label.config(text="Ошибка")
                messagebox.showerror(error_title, str(finished[1]))

        self.progress["value"] = 0
        self.status_label.config(text="Работаю...")
        threading.Thread(target=runner, daemon=True).start()
        self.after(40, poll)

    # ---------------- DECODE TAB ----------------
    def _build_decode_tab(self):
        top = ttk.Frame(self.decode_tab)
        top.pack(fill="x", pady=8, padx=8)

        ttk.Button(top, text="Открыть .ktx...", command=self.on_open_ktx).pack(side="left")
        self.decode_path_label = ttk.Label(top, text="Файл не выбран")
        self.decode_path_label.pack(side="left", padx=10)

        self.decode_info_label = ttk.Label(self.decode_tab, text="")
        self.decode_info_label.pack(anchor="w", padx=8)

        self.decode_canvas = tk.Label(self.decode_tab, text="Превью появится здесь",
                                      relief="groove", background="#222222", foreground="#aaaaaa")
        self.decode_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self.decode_tab)
        bottom.pack(fill="x", pady=8, padx=8)
        ttk.Button(bottom, text="Save as PNG...", command=self.on_save_png).pack(side="left")

    def on_open_ktx(self):
        path = filedialog.askopenfilename(
            title="Выбери .ktx файл",
            filetypes=[("KTX texture", "*.ktx"), ("Все файлы", "*.*")]
        )
        if not path:
            return

        def work(progress):
            progress(0.1, "Декодирование...")
            header, image_data = parse_ktx(path)
            check_simple_2d(header)
            w, h = header['pixelWidth'], header['pixelHeight']
            fmt_name, kind = FORMATS[header['glInternalFormat']]
            layers = layer_count(header)
            img = decode_layers(kind, image_data, w, h, layers)
            return img, w, h, fmt_name, len(image_data), header['numberOfMipmapLevels'], layers

        def done(res):
            img, w, h, fmt_name, nbytes, mips, layers = res
            self.decoded_image = img
            self.decode_source_path = path
            self.decode_path_label.config(text=path)
            self.decode_info_label.config(
                text=f"{w}x{h}  |  {fmt_name}  |  мип-уровней: {max(1, mips)}  |  "
                     f"{nbytes:,} байт данных изображения"
                     + (f"  |  слоёв: {layers} (лежат рядом слева направо)" if layers > 1 else ""))
            self._show_preview(self.decode_canvas, img)

        self._run_bg(work, done, "Ошибка декодирования")

    def on_save_png(self):
        if self.decoded_image is None:
            messagebox.showinfo("Нечего сохранять", "Сначала открой и декодируй .ktx файл.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить как PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return
        try:
            self.decoded_image.save(path)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n\n{e}")

    # ---------------- ENCODE TAB ----------------
    def _build_encode_tab(self):
        top = ttk.Frame(self.encode_tab)
        top.pack(fill="x", pady=8, padx=8)
        ttk.Button(top, text="1. Оригинальный .ktx (для формата)...",
                   command=self.on_pick_original_ktx).pack(side="left")
        self.encode_orig_label = ttk.Label(top, text="Не выбран")
        self.encode_orig_label.pack(side="left", padx=10)

        top2 = ttk.Frame(self.encode_tab)
        top2.pack(fill="x", pady=4, padx=8)
        ttk.Button(top2, text="2. Новая картинка (PNG/JPG)...",
                   command=self.on_pick_new_image).pack(side="left")
        self.encode_img_label = ttk.Label(top2, text="Не выбрана")
        self.encode_img_label.pack(side="left", padx=10)

        mid = ttk.Frame(self.encode_tab)
        mid.pack(fill="x", pady=8, padx=8)
        ttk.Button(mid, text="3. Encode (предпросмотр)", command=self.on_encode).pack(side="left")
        ttk.Label(mid, text="Размер:").pack(side="left", padx=(16, 4))
        self.encode_size_var = tk.StringVar(value="Как есть")
        ttk.Combobox(mid, textvariable=self.encode_size_var, values=list(SIZE_CHOICES),
                     state="readonly", width=12).pack(side="left")
        ttk.Label(mid, text="Фильтр:").pack(side="left", padx=(12, 4))
        self.encode_filter_var = tk.StringVar(value="Lanczos (плавно)")
        ttk.Combobox(mid, textvariable=self.encode_filter_var, values=list(FILTER_CHOICES),
                     state="readonly", width=24).pack(side="left")

        self.encode_info_label = ttk.Label(self.encode_tab, text="")
        self.encode_info_label.pack(anchor="w", padx=8)

        self.encode_canvas = tk.Label(self.encode_tab, text="Превью результата появится здесь",
                                      relief="groove", background="#222222", foreground="#aaaaaa")
        self.encode_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self.encode_tab)
        bottom.pack(fill="x", pady=8, padx=8)
        ttk.Button(bottom, text="4. Save as .ktx...", command=self.on_save_ktx).pack(side="left")

    def on_pick_original_ktx(self):
        path = filedialog.askopenfilename(
            title="Выбери оригинальный .ktx (берём оттуда формат/заголовок)",
            filetypes=[("KTX texture", "*.ktx"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        self.encode_original_ktx_path = path
        self.encode_orig_label.config(text=path)

    def on_pick_new_image(self):
        path = filedialog.askopenfilename(
            title="Выбери новую картинку",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            self.new_image = Image.open(path)
            self.new_image.load()
            self.new_image_path = path
            self.encode_img_label.config(
                text=f"{path}  ({self.new_image.size[0]}x{self.new_image.size[1]})")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть картинку:\n\n{e}")

    def on_encode(self):
        if not self.encode_original_ktx_path:
            messagebox.showinfo("Не хватает данных", "Сначала выбери оригинальный .ktx файл (шаг 1).")
            return
        if self.new_image is None:
            messagebox.showinfo("Не хватает данных", "Сначала выбери новую картинку (шаг 2).")
            return
        orig_path = self.encode_original_ktx_path
        image = self.new_image
        long_side = SIZE_CHOICES[self.encode_size_var.get()]
        resample = FILTER_CHOICES[self.encode_filter_var.get()]

        def work(progress):
            img = image.convert("RGBA")
            layers = ktx_layer_count(orig_path)
            lw = split_layers(img, layers)[0].width
            size = target_size(lw, img.height, long_side)
            if size != (lw, img.height):
                progress(0.02, f"Масштабирование до {size[0]}x{size[1]}...")
                img = resize_layers(img, layers, size, resample)
            res = build_ktx(img, orig_path,
                            lambda f, t: progress(0.1 + 0.8 * f, t))
            progress(0.92, "Готовлю превью...")
            # раскодируем обратно, чтобы честно показать, как это будет выглядеть в игре
            res["preview"] = decode_layers(res["kind"], res["mip0"], res["width"], res["height"],
                                           res["layers"])
            return res

        def done(res):
            self._encoded_bytes = res["bytes"]
            self.encode_info_label.config(text=self._result_text(res))
            self._show_preview(self.encode_canvas, res["preview"])

        self._run_bg(work, done, "Ошибка кодирования")

    def on_save_ktx(self):
        if self._encoded_bytes is None:
            messagebox.showinfo("Нечего сохранять", "Сначала нажми 'Encode' (шаг 3).")
            return
        self._save_ktx_dialog(self._encoded_bytes, None)

    # ---------------- UPSCALE TAB ----------------
    def _build_upscale_tab(self):
        top = ttk.Frame(self.upscale_tab)
        top.pack(fill="x", pady=8, padx=8)
        ttk.Button(top, text="1. Исходный .ktx...", command=self.on_pick_upscale_ktx).pack(side="left")
        self.upscale_src_label = ttk.Label(top, text="Не выбран")
        self.upscale_src_label.pack(side="left", padx=10)

        mid = ttk.Frame(self.upscale_tab)
        mid.pack(fill="x", pady=8, padx=8)
        ttk.Button(mid, text="2. Decode -> Upscale -> Encode",
                   command=self.on_upscale).pack(side="left")
        ttk.Label(mid, text="Размер:").pack(side="left", padx=(16, 4))
        self.upscale_size_var = tk.StringVar(value="8K (8192)")
        ttk.Combobox(mid, textvariable=self.upscale_size_var,
                     values=[k for k, v in SIZE_CHOICES.items() if v],
                     state="readonly", width=12).pack(side="left")
        ttk.Label(mid, text="Фильтр:").pack(side="left", padx=(12, 4))
        self.upscale_filter_var = tk.StringVar(value="Lanczos (плавно)")
        ttk.Combobox(mid, textvariable=self.upscale_filter_var, values=list(FILTER_CHOICES),
                     state="readonly", width=24).pack(side="left")

        self.upscale_info_label = ttk.Label(
            self.upscale_tab,
            text="Размер - это длинная сторона текстуры, пропорции сохраняются.")
        self.upscale_info_label.pack(anchor="w", padx=8)

        self.upscale_canvas = tk.Label(self.upscale_tab, text="Превью результата появится здесь",
                                       relief="groove", background="#222222", foreground="#aaaaaa")
        self.upscale_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self.upscale_tab)
        bottom.pack(fill="x", pady=8, padx=8)
        ttk.Button(bottom, text="3. Save as .ktx...", command=self.on_save_upscaled).pack(side="left")

    def on_pick_upscale_ktx(self):
        path = filedialog.askopenfilename(
            title="Выбери .ktx, который нужно увеличить",
            filetypes=[("KTX texture", "*.ktx"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        self.upscale_source_path = path
        self.upscale_src_label.config(text=path)

    def on_upscale(self):
        if not self.upscale_source_path:
            messagebox.showinfo("Не хватает данных", "Сначала выбери .ktx файл (шаг 1).")
            return
        src_path = self.upscale_source_path
        long_side = SIZE_CHOICES[self.upscale_size_var.get()]
        resample = FILTER_CHOICES[self.upscale_filter_var.get()]

        def work(progress):
            progress(0.01, "Декодирование...")
            header, image_data = parse_ktx(src_path)
            check_simple_2d(header)
            w, h = header['pixelWidth'], header['pixelHeight']
            _, kind = FORMATS[header['glInternalFormat']]
            layers = layer_count(header)
            img = decode_layers(kind, image_data, w, h, layers)
            size = target_size(w, h, long_side)
            progress(0.05, f"Масштабирование {w}x{h} -> {size[0]}x{size[1]}...")
            img = resize_layers(img, layers, size, resample)
            res = build_ktx(img, src_path,
                            lambda f, t: progress(0.15 + 0.75 * f, t))
            del img
            progress(0.92, "Готовлю превью...")
            res["preview"] = decode_layers(res["kind"], res["mip0"], res["width"], res["height"],
                                           res["layers"])
            res["source_size"] = (w, h)
            return res

        def done(res):
            self._upscaled_bytes = res["bytes"]
            sw, sh = res["source_size"]
            self.upscale_info_label.config(text=f"было {sw}x{sh}  ->  " + self._result_text(res))
            self._show_preview(self.upscale_canvas, res["preview"])

        self._run_bg(work, done, "Ошибка")

    def on_save_upscaled(self):
        if self._upscaled_bytes is None:
            messagebox.showinfo("Нечего сохранять", "Сначала нажми 'Decode -> Upscale -> Encode' (шаг 2).")
            return
        name = os.path.basename(self.upscale_source_path) if self.upscale_source_path else None
        self._save_ktx_dialog(self._upscaled_bytes, name)

    # ---------------- МАССОВЫЕ ВКЛАДКИ: общее ----------------
    def _load_settings(self):
        try:
            with open(settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            return
        for key, var in self._settings_vars.items():
            if key in data:
                try:
                    var.set(data[key])
                except Exception:  # noqa: BLE001
                    pass

    def _save_settings(self):
        data = {}
        for key, var in self._settings_vars.items():
            try:
                data[key] = var.get()
            except Exception:  # noqa: BLE001
                pass
        try:
            with open(settings_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        except Exception:  # noqa: BLE001
            pass

    def _path_row(self, parent, label, var, kind="dir"):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Обзор...", command=lambda: self._browse(var, kind)).pack(side="left", padx=(6, 0))

    def _browse(self, var, kind):
        if kind == "image":
            path = filedialog.askopenfilename(
                title="Выбери картинку",
                filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*.*")])
        else:
            path = filedialog.askdirectory(title="Выбери папку",
                                           initialdir=var.get() or os.path.expanduser("~"))
        if path:
            var.set(path)

    def _workers_box(self, parent):
        ttk.Label(parent, text="Потоков (0 = авто):").pack(side="left")
        ttk.Spinbox(parent, from_=0, to=worker_count(), width=4,
                    textvariable=self.var_workers).pack(side="left", padx=(4, 0))
        ttk.Label(parent, text=f"из {worker_count()}").pack(side="left", padx=(4, 0))

    def _get_workers(self):
        try:
            return max(0, int(self.var_workers.get().strip() or 0))
        except ValueError:
            return 0

    def _get_folders(self):
        res = []
        for slot, var in zip(SLOTS, (self.var_dirA, self.var_dirB)):
            path = var.get().strip()
            if not path:
                continue
            if not os.path.isdir(path):
                raise ValueError(f"Папка {slot} не найдена:\n{path}")
            res.append((slot, path))
        if not res:
            raise ValueError("Укажи хотя бы одну папку с .ktx (A или B).")
        return res

    def _make_log(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=8, pady=6)
        txt = tk.Text(frame, height=10, wrap="none", state="disabled",
                      background="#1e1e1e", foreground="#d4d4d4", font=("Consolas", 9))
        sb = ttk.Scrollbar(frame, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        return txt

    def _log(self, widget, line):
        """Можно звать из любого потока: строки забирает главный поток в _poll_log."""
        self._log_q.put((widget, line))

    def _poll_log(self):
        try:
            while True:
                widget, line = self._log_q.get_nowait()
                widget.configure(state="normal")
                widget.insert("end", line + "\n")
                if int(widget.index("end-1c").split(".")[0]) > 6000:
                    widget.delete("1.0", "1500.0")
                widget.see("end")
                widget.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _log_scan(self, widget, items):
        ok = [i for i in items if i["ok"]]
        bad = [i for i in items if not i["ok"]]
        self._log(widget, f"Найдено .ktx: {len(items)}  |  подходит: {len(ok)}  |  не подходит: {len(bad)}")
        for slot in SLOTS:
            part = [i for i in ok if i["slot"] == slot]
            if part:
                self._log(widget, f"  папка {slot} ({part[0]['tag']}): {len(part)} шт, "
                                  f"{fmt_size(sum(i['size'] for i in part))}")
        fmts = {}
        for i in ok:
            fmts[i["fmt"]] = fmts.get(i["fmt"], 0) + 1
        if fmts:
            self._log(widget, "  форматы: " + ", ".join(f"{k} x{v}" for k, v in sorted(fmts.items())))
            big = max(ok, key=lambda i: i["w"] * i["h"] * i["layers"])
            self._log(widget, f"  самая большая: {big['tag']}/{big['rel']}  {big['w']}x{big['h']}"
                              + (f" x{big['layers']} слоёв" if big['layers'] > 1 else ""))
        for i in bad[:20]:
            self._log(widget, f"  не подходит: {i['tag']}/{i['rel']}: {i['err']}")
        if len(bad) > 20:
            self._log(widget, f"  ...и ещё {len(bad) - 20}")

    def _log_summary(self, widget, summary, title):
        if summary is None:
            return
        tail = "  (ОСТАНОВЛЕНО)" if summary["cancelled"] else ""
        self._log(widget, f"=== {title}: готово {summary['ok']}, пропущено {summary['skip']}, "
                          f"ошибок {summary['err']} из {summary['total']} за {fmt_time(summary['elapsed'])}{tail}")

    def on_scan(self, widget):
        try:
            folders = self._get_folders()
        except ValueError as e:
            messagebox.showinfo("Укажи папки", str(e))
            return
        excl = [self.var_backup.get().strip(), self.var_decout.get().strip(), self.var_encout.get().strip()]
        self._save_settings()

        def work(progress):
            progress(0.1, "Сканирование папок...")
            return scan_ktx_folders(folders, excl)

        self._run_bg(work, lambda items: self._log_scan(widget, items), "Ошибка сканирования")

    # ---------------- МАССОВЫЙ DECODE ----------------
    def _build_bdec_tab(self):
        t = self.bdec_tab
        ttk.Label(t, text="Все .ktx из двух папок игры -> PNG. Подпапки обходятся рекурсивно, "
                          "файлы декодируются параллельно.").pack(anchor="w", padx=8, pady=(8, 4))
        self._path_row(t, "Папка текстур A:", self.var_dirA)
        self._path_row(t, "Папка текстур B:", self.var_dirB)
        self._path_row(t, "Выходная папка (PNG):", self.var_decout)
        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        self._workers_box(bar)
        ttk.Button(bar, text="Сканировать", command=lambda: self.on_scan(self.bdec_log)).pack(side="left", padx=(16, 0))
        ttk.Button(bar, text="Декодировать всё", command=self.on_bdecode).pack(side="left", padx=6)
        ttk.Button(bar, text="Отмена", command=self._cancel.set).pack(side="left")
        ttk.Label(t, text="Внутри выходной папки будут подпапки A_<имя> и B_<имя>, структура папок сохраняется. "
                          "У массивов слоёв слои лежат рядом слева направо.",
                  foreground="#777777").pack(anchor="w", padx=8)
        self.bdec_log = self._make_log(t)

    def on_bdecode(self):
        try:
            folders = self._get_folders()
        except ValueError as e:
            messagebox.showinfo("Укажи папки", str(e))
            return
        out = self.var_decout.get().strip()
        if not out:
            messagebox.showinfo("Укажи папку", "Выбери выходную папку для PNG.")
            return
        excl = [self.var_backup.get().strip(), out]
        req = self._get_workers()
        self._save_settings()
        self._cancel.clear()
        w = self.bdec_log

        def work(progress):
            progress(0.0, "Сканирование папок...")
            items = scan_ktx_folders(folders, excl)
            ok = [i for i in items if i["ok"]]
            self._log(w, f"Найдено .ktx: {len(items)}, декодируем: {len(ok)}, пропускаем: {len(items) - len(ok)}")
            for i in items:
                if not i["ok"]:
                    self._log(w, f"[SKIP] {i['tag']}/{i['rel']}  {i['err']}")
            if not ok:
                return None
            jobs = []
            for it in ok:
                job = dict(it)
                job.update(cost=it["w"] * it["h"] * it["layers"],
                           dst=os.path.join(out, it["tag"], os.path.splitext(it["rel"])[0] + ".png"))
                jobs.append(job)
            workers = req or auto_workers(max(j["cost"] for j in jobs))
            self._log(w, f"--- Decode: потоков {workers}" + (" (авто по RAM)" if not req else ""))
            return run_batch(_w_decode, jobs, workers, progress, lambda s: self._log(w, s), self._cancel, "Decode")

        self._run_bg(work, lambda s: self._log_summary(w, s, "Decode"), "Ошибка массового декодирования")

    # ---------------- МАССОВЫЙ ENCODE ----------------
    def _build_benc_tab(self):
        t = self.benc_tab
        self._path_row(t, "Папка текстур A:", self.var_dirA)
        self._path_row(t, "Папка текстур B:", self.var_dirB)

        dest = ttk.LabelFrame(t, text="Куда писать результат")
        dest.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Radiobutton(dest, text="Обратно в папки игры (поверх оригиналов)",
                        variable=self.var_target, value="game").pack(anchor="w", padx=8)
        ttk.Checkbutton(dest, text="Сначала сохранить оригиналы (бэкап, один раз - повторный запуск не затирает)",
                        variable=self.var_do_backup).pack(anchor="w", padx=28)
        self._path_row(dest, "Папка бэкапов:", self.var_backup)
        ttk.Radiobutton(dest, text="В отдельную папку (игра не трогается)",
                        variable=self.var_target, value="folder").pack(anchor="w", padx=8)
        self._path_row(dest, "Отдельная папка:", self.var_encout)

        mode = ttk.LabelFrame(t, text="Режим")
        mode.pack(fill="x", padx=8, pady=2)
        ttk.Radiobutton(mode, text="1. Перекодировать ВСЕ в разрешение (декод -> растяжение -> энкод)",
                        variable=self.var_mode, value="rescale").pack(anchor="w", padx=8)
        ttk.Radiobutton(mode, text="2. Одна картинка -> во все .ktx (формат, слои и мипы берутся у каждого оригинала)",
                        variable=self.var_mode, value="single").pack(anchor="w", padx=8)
        self._path_row(mode, "    Картинка для режима 2:", self.var_single_img, kind="image")
        ttk.Radiobutton(mode, text="3. Режим изучения: на каждой текстуре её номер + таблица номеров",
                        variable=self.var_mode, value="number").pack(anchor="w", padx=8)
        self._path_row(mode, "    Папка для таблицы номеров:", self.var_logdir)

        opts = ttk.Frame(t)
        opts.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(opts, text="Размер (длинная сторона):").pack(side="left")
        ttk.Combobox(opts, textvariable=self.var_bsize, values=list(SIZE_CHOICES),
                     state="readonly", width=12).pack(side="left", padx=(4, 12))
        ttk.Label(opts, text="Фильтр:").pack(side="left")
        ttk.Combobox(opts, textvariable=self.var_bfilter, values=list(FILTER_CHOICES),
                     state="readonly", width=24).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(opts, text="Пропускать, если уже такого размера или больше",
                        variable=self.var_skip_ge).pack(side="left")
        opts2 = ttk.Frame(t)
        opts2.pack(fill="x", padx=8, pady=2)
        ttk.Label(opts2, text="Режим 1: брать только текстуры с длинной стороной не меньше:").pack(side="left")
        ttk.Combobox(opts2, textvariable=self.var_minsrc, values=["0", "64", "128", "256", "512", "1024"],
                     width=6).pack(side="left", padx=(4, 0))
        ttk.Label(opts2, text="(0 = все; мелкие иконки обычно растягивать незачем)",
                  foreground="#777777").pack(side="left", padx=6)

        bar = ttk.Frame(t)
        bar.pack(fill="x", padx=8, pady=6)
        self._workers_box(bar)
        ttk.Button(bar, text="Сканировать", command=lambda: self.on_scan(self.benc_log)).pack(side="left", padx=(16, 0))
        ttk.Button(bar, text="Запустить", command=self.on_bencode).pack(side="left", padx=6)
        ttk.Button(bar, text="Отмена", command=self._cancel.set).pack(side="left")
        ttk.Button(bar, text="Восстановить из бэкапа", command=self.on_restore).pack(side="left", padx=(16, 0))
        self.benc_log = self._make_log(t)

    def _log_plan(self, widget, plan):
        self._log(widget, f"План: к обработке {len(plan['jobs'])}, пропущено по фильтрам {len(plan['skipped'])}, "
                          f"не подходит {len(plan['bad'])}")
        for it, why in plan["skipped"][:10]:
            self._log(widget, f"  [SKIP] {it['tag']}/{it['rel']}  {why}")
        if len(plan["skipped"]) > 10:
            self._log(widget, f"  ...и ещё {len(plan['skipped']) - 10} пропущено")
        for it in plan["bad"][:10]:
            self._log(widget, f"  [SKIP] {it['tag']}/{it['rel']}  {it['err']}")
        if len(plan["bad"]) > 10:
            self._log(widget, f"  ...и ещё {len(plan['bad']) - 10} не подходит")

    def on_bencode(self):
        try:
            folders = self._get_folders()
        except ValueError as e:
            messagebox.showinfo("Укажи папки", str(e))
            return
        mode = self.var_mode.get()
        to_game = self.var_target.get() == "game"
        out_dir = "" if to_game else self.var_encout.get().strip()
        backup_dir = self.var_backup.get().strip() if (to_game and self.var_do_backup.get()) else ""
        if not to_game and not out_dir:
            messagebox.showinfo("Укажи папку", "Выбери отдельную папку для результата.")
            return
        if to_game and self.var_do_backup.get() and not backup_dir:
            messagebox.showinfo("Укажи папку", "Укажи папку для бэкапов оригиналов.")
            return
        long_side = SIZE_CHOICES.get(self.var_bsize.get())
        if mode == "rescale" and not long_side:
            messagebox.showinfo("Укажи размер", "Для режима 1 выбери 2K / 4K / 8K.")
            return
        image_path = self.var_single_img.get().strip()
        if mode == "single" and not os.path.isfile(image_path):
            messagebox.showinfo("Укажи картинку", "Для режима 2 выбери картинку.")
            return
        try:
            min_src = int(self.var_minsrc.get().strip() or 0)
        except ValueError:
            min_src = 0
        opts = dict(long_side=long_side, skip_ge=self.var_skip_ge.get(), min_src=min_src,
                    out_dir=out_dir, backup_dir=backup_dir, filter=self.var_bfilter.get(),
                    image=image_path)
        excl = [self.var_backup.get().strip(), self.var_decout.get().strip(), out_dir]
        self._save_settings()
        w = self.benc_log

        def scan_work(progress):
            progress(0.1, "Сканирование папок...")
            return scan_ktx_folders(folders, excl)

        def after_scan(items):
            try:
                self._confirm_and_run(items, mode, opts, to_game, w)
            except Exception as e:  # noqa: BLE001
                messagebox.showerror("Ошибка", str(e))

        self._run_bg(scan_work, after_scan, "Ошибка сканирования")

    def _confirm_and_run(self, items, mode, opts, to_game, w):
        plan = plan_encode_jobs(items, mode, opts)
        self._log_plan(w, plan)
        jobs = plan["jobs"]
        if not jobs:
            messagebox.showinfo("Нечего делать", "Подходящих файлов не нашлось (или всё отфильтровано).")
            return
        mode_name = {"rescale": f"перекодирование в {self.var_bsize.get()}",
                     "single": "одна картинка во все .ktx",
                     "number": "режим изучения (номера на текстурах)"}[mode]
        lines = [f"Режим: {mode_name}",
                 f"Файлов к обработке: {len(jobs)}  (пропущено: {len(plan['skipped'])}, "
                 f"не подходит: {len(plan['bad'])})",
                 f"Новых данных примерно {fmt_size(plan['est_new'])} "
                 f"(сейчас эти файлы занимают {fmt_size(plan['est_old'])})", ""]
        need = []
        if to_game:
            lines.append("Файлы будут ПЕРЕЗАПИСАНЫ прямо в папках игры.")
            if opts["backup_dir"]:
                lines.append(f"Бэкап оригиналов: {opts['backup_dir']}  ({fmt_size(plan['est_old'])})")
                need.append((opts["backup_dir"], plan["est_old"], "бэкапа"))
            else:
                lines.append("БЭКАП ВЫКЛЮЧЕН - вернуть оригиналы будет нечем!")
            need.append((jobs[0]["root"], max(0, plan["est_new"] - plan["est_old"]), "игры"))
        else:
            lines.append(f"Результат будет записан в: {opts['out_dir']}")
            need.append((opts["out_dir"], plan["est_new"], "результата"))
        for path, size, what in need:
            free = free_bytes(path)
            if free is not None and size * 1.05 > free:
                lines.append(f"ВНИМАНИЕ: на диске {what} не хватает места "
                             f"(нужно ~{fmt_size(size)}, свободно {fmt_size(free)})")
        if not messagebox.askyesno("Подтверждение", "\n".join(lines) + "\n\nПродолжить?", icon="warning"):
            self._log(w, "Отменено пользователем до начала.")
            return
        self._start_encode(plan, mode, opts, w)

    def _start_encode(self, plan, mode, opts, w):
        jobs = plan["jobs"]
        worker = WORKERS[mode]
        label = {"rescale": "Encode", "single": "Одна картинка", "number": "Нумерация"}[mode]
        max_px = max(j["cost"] for j in jobs)
        if mode == "single":
            try:
                with Image.open(opts["image"]) as im:
                    max_px = max(max_px, im.width * im.height)
            except Exception:  # noqa: BLE001
                pass
        req = self._get_workers()
        logdir = self.var_logdir.get().strip() or app_dir()
        self._cancel.clear()

        def work(progress):
            workers = req or auto_workers(max_px)
            self._log(w, f"--- {label}: {len(jobs)} файлов, потоков {workers}" + (" (авто по RAM)" if not req else ""))
            log_path = None
            if mode == "number":
                os.makedirs(logdir, exist_ok=True)
                log_path = os.path.join(logdir, time.strftime("ktx_numbering_%Y%m%d_%H%M%S.csv"))
                write_numbering_log(log_path, jobs, [])        # заранее: пригодится, даже если прервут
            summary = run_batch(worker, jobs, workers, progress, lambda s: self._log(w, s), self._cancel, label)
            if log_path:
                write_numbering_log(log_path, jobs, summary["results"])
            summary["log_path"] = log_path
            return summary

        def done(summary):
            self._log_summary(w, summary, label)
            if mode == "number" and summary:
                self._log(w, "--- Номер -> текстура ---")
                for j in sorted(jobs, key=lambda j: j["number"]):
                    self._log(w, f"#{j['number']:<5} {j['tag']}/{j['rel']}   {j['w']}x{j['h']}  {j['fmt']}"
                                 + (f"  слоёв {j['layers']}" if j["layers"] > 1 else ""))
                self._log(w, f"Таблица сохранена: {summary['log_path']}")

        self._run_bg(work, done, "Ошибка массового кодирования")

    def on_restore(self):
        try:
            folders = self._get_folders()
        except ValueError as e:
            messagebox.showinfo("Укажи папки", str(e))
            return
        backup_dir = self.var_backup.get().strip()
        if not backup_dir or not os.path.isdir(backup_dir):
            messagebox.showinfo("Нет бэкапа", "Папка бэкапов не найдена.")
            return
        pairs = plan_restore(folders, backup_dir)
        if not pairs:
            messagebox.showinfo("Нет бэкапа", "В папке бэкапов нет файлов для выбранных папок A/B.\n"
                                              "(Подпапки бэкапа называются A_<имя папки> и B_<имя папки>.)")
            return
        if not messagebox.askyesno("Восстановление",
                                   f"Вернуть {len(pairs)} оригинальных файлов из бэкапа в папки игры?\n"
                                   "Текущие версии этих файлов будут заменены.", icon="warning"):
            return
        w = self.benc_log
        self._cancel.clear()

        def work(progress):
            self._log(w, f"--- Восстановление: {len(pairs)} файлов")
            return restore_files(pairs, progress, lambda s: self._log(w, s), self._cancel)

        def done(res):
            ok, err = res
            self._log(w, f"=== Восстановлено {ok}, ошибок {err}")

        self._run_bg(work, done, "Ошибка восстановления")

    # ---------------- общие утилиты ----------------
    @staticmethod
    def _result_text(res):
        layers = f"слоёв: {res['layers']}  |  " if res.get("layers", 1) > 1 else ""
        return (f"{res['width']}x{res['height']}  |  {res['fmt_name']}  |  {layers}"
                f"мип-уровней: {res['levels']}  |  {len(res['bytes']):,} байт  |  "
                "превью - результат ПОСЛЕ сжатия")

    def _save_ktx_dialog(self, data, initial_name):
        kwargs = dict(title="Сохранить как .ktx", defaultextension=".ktx",
                      filetypes=[("KTX texture", "*.ktx")])
        if initial_name:
            kwargs["initialfile"] = initial_name
        path = filedialog.asksaveasfilename(**kwargs)
        if not path:
            return
        try:
            with open(path, "wb") as f:
                f.write(data)
            messagebox.showinfo("Готово", f"Сохранено:\n{path}\n\n"
                                          "Ты сам решаешь, куда положить этот файл дальше.")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n\n{e}")

    def _show_preview(self, label_widget, pil_image):
        w, h = pil_image.size
        scale = min(1.0, 800 / w, 450 / h)
        if scale < 1.0:
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = pil_image.resize(size, _RES.BILINEAR, reducing_gap=2.0)
        else:
            img = pil_image
        tk_img = ImageTk.PhotoImage(img)
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img  # держим ссылку, иначе Tkinter соберёт мусор


if __name__ == "__main__":
    multiprocessing.freeze_support()   # нужно, если собирать в .exe
    app = KtxToolApp()
    app.mainloop()
