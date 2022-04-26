from collections.abc import MutableMapping
from datetime import datetime
    
class AccumulativeDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __str__(self):
        res_str = ''
        for k, v in self.items():
            res_str += f'{k}: {v:.5f};\n'
        return res_str

    def __add__(self, value):
        result = dict()  # create resulting object
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            # if first term is empty
            if not self.keys():
                return AccumulativeDict(value)  # return copy of second term as a resulting object
            # else, get overall keys
            k_overall = self.keys() | value.keys()
            # foreach key in the overall collection
            for k in k_overall:
                a = self[k] if k in self.keys() else 0  # take first arg-of-sum
                b = value[k] if k in value.keys() else 0  # take second arg-of-sum
                result[k] = a + b  # accumulate them in the resulting object
        else:
            for k, v in self.items():
                result[k] = self[k] + value
        return AccumulativeDict(result)

    def __neg__(self):
        result = dict()
        for k, v in self.items():
            result[k] = -v
        return result

    def __sub__(self, value):
        result = dict()  # create resulting object
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            # if first term is empty
            if not self.keys():
                return -AccumulativeDict(value)  # return copy -second term as a resulting object
            # else, get intersection of keys
            k_intersect = self.keys() & value.keys()
            # foreach term in intersection
            for k in k_intersect:
                result[k] = self[k] - value[k]  # save the difference as for resulting object
        else:
            for k, v in self.items():
                result[k] = self[k] - value
        return AccumulativeDict(result)

    def __mul__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] * value[k]
        else:
            for k, v in self.items():
                result[k] = self[k] * value
        return AccumulativeDict(result)

    def __floordiv__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] // value[k]
        else:
            for k, v in self.items():
                result[k] = self[k] // value
        return AccumulativeDict(result)

    def __truediv__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] / value[k]
        else:
            for k, v in self.items():
                result[k] = self[k] / value
        return AccumulativeDict(result)
    
def res_to_string(results, epoch, epoch_max):
    res_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f': Epoch {epoch}/{epoch_max},'
    for name, value in results.items():
        res_str+=f"{name}: {value:.5f};"
    return res_str

def plot_params(log_data, writer, epoch):
    for name,value in log_data.items():
        writer.add_scalar(name, value, epoch)