class SOLVED_ACCURACY():
    def __init__(self):
        self.reset()

    def reset(self):
        self.solved = 0
        self.count = 0
    
    def update(self, solved_capacity, ground_truth_capacity, batch_size=1):
        if solved_capacity == ground_truth_capacity:
            self.solved += batch_size
        self.count += batch_size
    
    def accuracy(self):
        return self.solved / self.count

class AVG_SIZE():
    def __init__(self):
        self.reset()

    def reset(self):
        self.size = 0
        self.count = 0
        self.avg = 0
    
    def update(self, size, batch_size=1):
        self.size += size
        self.count += batch_size
        if self.count > 0:
            self.avg = self.size / self.count

class AVG_TIME():
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.time = 0
        self.count = 0

    def update(self, time, batch_size=1):
        self.time += time
        self.count += batch_size

    def avg_time(self):
        return self.time / self.count