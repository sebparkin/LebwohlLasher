#Compare times for each approach that is able to use default python
import numpy as np
import pandas as pd

from LL_numba import LL_numba_timing
from LL_numpy import LL_numpy_timing
from LL_cython import LL_cython_timing
import LebwohlLasher_timing

params = [200, 50, 0.5]
tot_time = np.zeros(4)

for i in range(3):
    tot_time[0] += LebwohlLasher_timing.main(200, 50, 0.5)
    tot_time[1] += LL_numpy_timing.main(200, 50, 0.5)
    tot_time[2] += LL_numba_timing.main(200, 50, 0.5)
    tot_time[3] += LL_cython_timing.main(200, 50, 0.5)

avg_time = tot_time / 3

index = ['Default', 'NumPy', 'Numba', 'Cython']

df = pd.DataFrame(data=avg_time, index=index)

df.to_csv('./times/compare_python.csv')
