from typing import Union


def progress(idx: int, max_idx: int, *, desc: str = Union[str, None], b_len: int = 50) -> None:
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
        # print progress bar
        print(f"{p_bar}  {desc}", end='')
        return None


def convert_time(seconds: Union[float, int]) -> str:
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
