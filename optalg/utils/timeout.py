import signal


class Timeout:
    """

    Ref:
    ---
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish

    Usage:
    ----
    with Timeout(seconds=2):
        time.sleep(4)

    Note
    ---
    Requires import signal, only works in linux


    If timeout, it raises error: raise TimeoutError
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def run_with_timeout(fun, seconds=5):
    with Timeout(seconds=seconds):
        try:
            return fun()
        except BaseException:
            pass
