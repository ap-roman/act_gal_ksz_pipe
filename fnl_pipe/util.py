import time


# NOT thread safe
class Timer:
    def __init__(self):
        self.dt = 0.
        self.last_t = None
        self.active = False

    def start(self):
        if not self.active:
            self.last_t = time.time()
            self.active = True

    def stop(self):
        self.dt += time.time() - self.last_t
        self.active = False