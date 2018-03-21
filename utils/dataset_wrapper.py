class DatasetWrapper(object):
    def __init__(self, dataset, count):
        self.count = count
        self.idx = 0
        self.dataset = dataset
        self.iter_dataset = iter(dataset)
        self.epoch = 0

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.count:
            self.idx = 0
            raise StopIteration

        self.idx += 1
        while True:
            try:
                return next(self.iter_dataset)
            except StopIteration:
                self.iter_dataset = iter(self.dataset)
                self.epoch += 1
                try:
                    return next(self.iter_dataset)
                except StopIteration:
                    raise Exception("Appears as if dataset is empty")
