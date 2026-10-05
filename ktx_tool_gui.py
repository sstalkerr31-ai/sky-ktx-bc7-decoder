#!/usr/bin/env python3
"""
KTX Texture Tool - простое GUI для личного использования
============================================================
Decode:  выбрал .ktx -> увидел превью -> сам решил, сохранять ли PNG и куда.
Encode:  выбрал новую картинку + оригинальный .ktx (для формата/заголовка)
         -> (по желанию растянул до 2K/4K/8K) -> Save As.
Upscale: выбрал .ktx -> декод -> растяжение до 8K -> энкод -> Save As.

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
import math
import multiprocessing
import os
import queue
import struct
import threading
import time
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

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
    if header['numberOfFaces'] > 1 or header['numberOfArrayElements'] > 0:
        raise ValueError("Этот .ktx содержит кубмапу/массив слоёв.\n"
                         "Инструмент поддерживает только простые 2D-текстуры.")


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


def encode_level(img, kind, on_blocks=lambda n: None):
    """Кодирует один мип-уровень. on_blocks(n) вызывается по мере готовности блоков."""
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

    if bh * bw < _PARALLEL_MIN_BLOCKS or worker_count() == 1:
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


def build_ktx(image, original_ktx_path, progress=lambda frac, text: None):
    """Собирает новый .ktx с заголовком оригинала и картинкой image.

    Возвращает dict: bytes, mip0 (сжатые данные верхнего уровня), kind,
    fmt_name, width, height, levels."""
    with open(original_ktx_path, "rb") as f:
        orig = f.read()
    header_d, _, header_len = parse_ktx_bytes(orig)
    check_simple_2d(header_d)
    fmt_name, kind = FORMATS[header_d['glInternalFormat']]

    img = image.convert("L" if kind == "bc4" else "RGBA")
    w, h = img.size
    levels = mip_level_count(header_d, w, h)

    sizes = [(max(1, w >> i), max(1, h >> i)) for i in range(levels)]
    total_blocks = sum(((mw + 3) // 4) * ((mh + 3) // 4) for mw, mh in sizes)
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
    level_img = img
    for i, (mw, mh) in enumerate(sizes):
        if i > 0:
            level_img = resize_image(level_img, (mw, mh), _RES.BOX)
        data = encode_level(level_img, kind, on_blocks)
        if i == 0:
            mip0 = data
        out.append(struct.pack("<I", len(data)))
        out.append(data)
        if len(data) % 4:
            out.append(b"\x00" * (4 - len(data) % 4))

    return dict(bytes=b"".join(out), mip0=mip0, kind=kind, fmt_name=fmt_name,
                width=w, height=h, levels=levels)


def repack_ktx_bytes(new_image, original_ktx_path):
    """Совместимость со старым именем функции."""
    return build_ktx(new_image, original_ktx_path)["bytes"]


# ============================================================
#  GUI
# ============================================================

class KtxToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KTX Texture Tool - для личного использования")
        self.geometry("960x720")
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

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        self.decode_tab = ttk.Frame(notebook)
        self.encode_tab = ttk.Frame(notebook)
        self.upscale_tab = ttk.Frame(notebook)
        notebook.add(self.decode_tab, text="Decode (.ktx -> PNG)")
        notebook.add(self.encode_tab, text="Encode (PNG -> .ktx)")
        notebook.add(self.upscale_tab, text="Upscale 8K (.ktx -> .ktx)")

        self._build_decode_tab()
        self._build_encode_tab()
        self._build_upscale_tab()

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8, pady=(0, 8))
        self.progress = ttk.Progressbar(status, maximum=1000, length=260)
        self.progress.pack(side="left")
        self.status_label = ttk.Label(status, text=f"Готов. Ядер для кодирования: {worker_count()}")
        self.status_label.pack(side="left", padx=10)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
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
            img = decode_image(kind, image_data, w, h)
            return img, w, h, fmt_name, len(image_data), header['numberOfMipmapLevels']

        def done(res):
            img, w, h, fmt_name, nbytes, mips = res
            self.decoded_image = img
            self.decode_source_path = path
            self.decode_path_label.config(text=path)
            self.decode_info_label.config(
                text=f"{w}x{h}  |  {fmt_name}  |  мип-уровней: {max(1, mips)}  |  "
                     f"{nbytes:,} байт данных изображения")
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
            size = target_size(img.width, img.height, long_side)
            if size != img.size:
                progress(0.02, f"Масштабирование до {size[0]}x{size[1]}...")
                img = resize_image(img, size, resample)
            res = build_ktx(img, orig_path,
                            lambda f, t: progress(0.1 + 0.8 * f, t))
            progress(0.92, "Готовлю превью...")
            # раскодируем обратно, чтобы честно показать, как это будет выглядеть в игре
            res["preview"] = decode_image(res["kind"], res["mip0"], res["width"], res["height"])
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
            img = decode_image(kind, image_data, w, h)
            size = target_size(w, h, long_side)
            progress(0.05, f"Масштабирование {w}x{h} -> {size[0]}x{size[1]}...")
            img = resize_image(img, size, resample)
            res = build_ktx(img, src_path,
                            lambda f, t: progress(0.15 + 0.75 * f, t))
            del img
            progress(0.92, "Готовлю превью...")
            res["preview"] = decode_image(res["kind"], res["mip0"], res["width"], res["height"])
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

    # ---------------- общие утилиты ----------------
    @staticmethod
    def _result_text(res):
        return (f"{res['width']}x{res['height']}  |  {res['fmt_name']}  |  "
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
