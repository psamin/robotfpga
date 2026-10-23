"""CMA buffers and MMIO for the KR260 engine without PYNQ: ctypes on XRT's libxrt_core and /dev/mem.

PYNQ 3.0.1 only runs on the board's Python 3.10, while LeRobot needs 3.12. The engine itself needs just
two things from PYNQ: physically contiguous DMA buffers (XRT buffer objects on the zocl driver, which is
what pynq.buffer uses underneath) and the IP's AXI-Lite registers (/dev/mem). This module gives both to
any Python, so the policy process can drive the FPGA directly instead of copying activations through a
socket. Needs root. The bitstream is still loaded once with PYNQ (engine_server.py or load_overlay.py);
it stays programmed after that process exits.

    dev = XrtDevice()
    buf = dev.allocate(1 << 20)         # cacheable CMA; buf.array is a uint8 numpy view
    buf.flush(0, nbytes); buf.invalidate(0, nbytes); buf.physical_address
    regs = Mmio(0xA0000000, 0x10000); regs.write(0x00, 1); regs.read(0x00)
"""

from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

XCL_BO_DEVICE_RAM = 3
XCL_BO_SYNC_BO_TO_DEVICE = 0
XCL_BO_SYNC_BO_FROM_DEVICE = 1
ZOCL_BO_FLAGS_CACHEABLE = 1 << 24


class _BOProperties(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("size", ctypes.c_ulonglong),
        ("paddr", ctypes.c_ulonglong),
        ("domain", ctypes.c_uint),
    ]


class XrtBuffer:
    def __init__(self, dev: XrtDevice, bo: int, nbytes: int):
        self.dev, self.bo, self.nbytes = dev, bo, nbytes
        lib = dev.lib
        ptr = lib.xclMapBO(dev.handle, bo, True)
        if not ptr:
            raise RuntimeError("xclMapBO failed")
        self.ptr = ptr
        self.array = np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(ptr))
        prop = _BOProperties()
        if lib.xclGetBOProperties(dev.handle, bo, ctypes.byref(prop)):
            raise RuntimeError("xclGetBOProperties failed")
        self.physical_address = int(prop.paddr)

    def view(self, dtype, offset: int = 0, count: int | None = None) -> np.ndarray:
        a = self.array[offset:].view(dtype)
        return a if count is None else a[:count]

    def _sync(self, direction: int, offset: int, nbytes: int) -> None:
        r = self.dev.lib.xclSyncBO(self.dev.handle, self.bo, direction, nbytes, offset)
        if r:
            raise RuntimeError(f"xclSyncBO failed: {r}")

    def flush(self, offset: int = 0, nbytes: int | None = None) -> None:
        """CPU cache -> DDR for [offset, offset + nbytes); PynqBuffer.flush always does the whole buffer."""
        self._sync(XCL_BO_SYNC_BO_TO_DEVICE, offset, self.nbytes - offset if nbytes is None else nbytes)

    def invalidate(self, offset: int = 0, nbytes: int | None = None) -> None:
        self._sync(XCL_BO_SYNC_BO_FROM_DEVICE, offset, self.nbytes - offset if nbytes is None else nbytes)

    def free(self) -> None:
        if self.bo is not None:
            self.array = None
            libc = ctypes.CDLL(None)
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munmap(self.ptr, self.nbytes)
            self.dev.lib.xclFreeBO(self.dev.handle, self.bo)
            self.bo = None


class XrtDevice:
    def __init__(self, index: int = 0, lib_path: str = "libxrt_core.so.2"):
        lib = ctypes.CDLL(lib_path)
        lib.xclOpen.restype = ctypes.c_void_p
        lib.xclOpen.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]
        lib.xclClose.argtypes = [ctypes.c_void_p]
        lib.xclAllocBO.restype = ctypes.c_uint
        lib.xclAllocBO.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint]
        lib.xclFreeBO.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        lib.xclMapBO.restype = ctypes.c_void_p
        lib.xclMapBO.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_bool]
        lib.xclSyncBO.restype = ctypes.c_int
        lib.xclSyncBO.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.xclGetBOProperties.restype = ctypes.c_int
        lib.xclGetBOProperties.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(_BOProperties)]
        self.lib = lib
        self.handle = lib.xclOpen(index, None, 0)
        if not self.handle:
            raise RuntimeError("xclOpen failed (run as root; is the zocl driver loaded?)")

    def allocate(self, nbytes: int, cacheable: bool = True) -> XrtBuffer:
        flags = ZOCL_BO_FLAGS_CACHEABLE if cacheable else 0
        bo = self.lib.xclAllocBO(self.handle, nbytes, XCL_BO_DEVICE_RAM, flags)
        if bo >= 0x80000000:
            raise MemoryError(f"xclAllocBO({nbytes}) failed: CMA exhausted? (grep Cma /proc/meminfo)")
        return XrtBuffer(self, bo, nbytes)


class Mmio:
    """A page-aligned /dev/mem window with 32-bit register access."""

    def __init__(self, base: int, length: int):
        self.base = base
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        try:
            self.mem = mmap.mmap(fd, length, offset=base)
        finally:
            os.close(fd)
        self.regs = np.frombuffer(self.mem, dtype=np.uint32)

    def read(self, offset: int) -> int:
        return int(self.regs[offset >> 2])

    def write(self, offset: int, value: int) -> None:
        self.regs[offset >> 2] = value & 0xFFFFFFFF

    def write64(self, offset: int, value: int) -> None:
        self.regs[offset >> 2] = value & 0xFFFFFFFF
        self.regs[(offset >> 2) + 1] = (value >> 32) & 0xFFFFFFFF
