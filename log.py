from contextlib import contextmanager

_prefix = None
_enabled = False
_throttle = False
_buffer = []
@contextmanager
def prefix(prefix):
    global _prefix
    _prefix = prefix
    try:
        yield
    except Exception as e:
        flush_buffer()
        raise e
    finally:
        _prefix = None
        flush_buffer()


def output(prefix, args, kwargs):
    if (prefix is not None):
        print(prefix, end="")
    print(*args, **kwargs)

def flush_buffer():
    global _buffer
    for b in _buffer:
        output(*b)
    _buffer = []

def log(*args, **kwargs):
    global _enabled
    if not _enabled: return


    global _prefix, _throttle, _buffer
    if _throttle:
        _buffer.append([_prefix, args, kwargs])
        if len(_buffer) > 30:
            flush_buffer()
    else:
        output(_prefix, args, kwargs)


def enable():
    global _enabled
    _enabled = True
def throttle():
    global _throttle
    _throttle = True