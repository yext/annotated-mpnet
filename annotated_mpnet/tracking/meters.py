"""
Helper class borrowed from fairseq to help store values as we train across multiple steps and epochs
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all stored values to 0
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the average appropriately for the meter
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
