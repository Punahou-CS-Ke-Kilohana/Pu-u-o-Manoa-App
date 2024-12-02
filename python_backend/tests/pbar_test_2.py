import time


def progress(idx: int, max_idx: int, *, desc: str = None, b_len: int = 50) -> None:
    assert isinstance(b_len, int), "'b_len' is not an integer"
    # completed progress
    completed = (idx + 1) / max_idx
    # make progress bar
    p_bar = (
        f"\r[{'—' * int(b_len * completed)}"
        f"{' ' * (b_len - int(b_len * completed))}]"
    )
    if desc is None:
        # print progress bar
        print(p_bar, end='')
        return None
    else:
        # print description
        print(f"{p_bar}  {desc}", end='')
        return None


def convert_time(seconds: float) -> str:
    # round seconds
    seconds = int(seconds)
    # find minutes and hours
    minutes = seconds // 60
    hours = minutes // 60
    # adjust times
    minutes -= 60 * hours
    seconds -= 60 * minutes + 3600 * hours
    # return time
    return f"{hours:01}:{minutes:02}:{seconds:02}"


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
    printed = True
