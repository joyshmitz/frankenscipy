#!/usr/bin/env python3
"""Oracle: numpy.loadtxt / savetxt on a 500x20 matrix, matching fsci-io bench_text_io
(crates/fsci-io/benches/io_bench.rs: matrix(rows,cols)=i*0.001+1.0, space delimiter).
"""
import time, io
import numpy as np


def med(fn, r=9):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


if __name__ == "__main__":
    rows, cols = 500, 20
    data = np.array([i * 0.001 + 1.0 for i in range(rows * cols)]).reshape(rows, cols)
    buf = io.StringIO(); np.savetxt(buf, data, delimiter=" "); text = buf.getvalue()
    def save():
        b = io.StringIO(); np.savetxt(b, data, delimiter=" ")
    def load():
        np.loadtxt(io.StringIO(text))
    print(f"numpy savetxt 500x20: {med(save)*1e6:.2f} us")
    print(f"numpy loadtxt 500x20: {med(load)*1e6:.2f} us")
