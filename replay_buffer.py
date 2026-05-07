import numpy as np

class ReplayBuffer:
    def __init__(self, size=100000):
        self.size = size
        self.ptr = 0
        self.full = False

        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.d = []

    def add(self, s, a, r, s2, done):
        if len(self.s) < self.size:
            self.s.append(s)
            self.a.append(a)
            self.r.append(r)
            self.s2.append(s2)
            self.d.append(done)
        else:
            idx = self.ptr
            self.s[idx] = s
            self.a[idx] = a
            self.r[idx] = r
            self.s2[idx] = s2
            self.d[idx] = done

        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch_size=64):
        idx = np.random.randint(0, len(self.s), size=batch_size)
        return (
            np.array([self.s[i] for i in idx]),
            np.array([self.a[i] for i in idx]),
            np.array([self.r[i] for i in idx]),
            np.array([self.s2[i] for i in idx]),
            np.array([self.d[i] for i in idx]),
        )