
class UpdatingAverage:
    """
    record the float value and return the average of records
    """

    def __init__(self):
        self.steps: int   = 0
        self.sum:   float = 0

    def update(self, val):
        self.sum   += val
        self.steps += 1

    def __call__(self, *args, **kwargs):
        return self.sum / float(self.steps)
