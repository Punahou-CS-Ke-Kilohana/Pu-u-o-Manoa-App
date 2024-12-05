from application import progress, convert_time
import time

max_iter = 100
loss = 0.1

start = time.time()

for i in range(max_iter):
    time.sleep(0.1)
    elapsed = time.time() - start
    desc = (
        f"{str(i + 1).zfill(len(str(max_iter)))}it/{max_iter}it  "
        f"{(100 * (i + 1) / max_iter):05.1f}%  "
        f"{loss:.3}loss  "
        f"{convert_time(elapsed)}et  "
        f"{convert_time(elapsed * max_iter / (i + 1) - elapsed)}eta  "
        f"{round((i + 1) / elapsed, 1)}it/s"
    )
    progress(i, max_iter, desc=desc)
