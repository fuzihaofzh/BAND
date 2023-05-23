#!/usr/bin/env python
import GPUtil
import numpy as np
if __name__ == '__main__':
    gpus = GPUtil.getGPUs()
    print(np.argmax(list(map(lambda x: x.memoryFree, gpus))))
