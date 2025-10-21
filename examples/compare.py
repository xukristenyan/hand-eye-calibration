import numpy as np

T_new = np.load('/home/necl/Projects/hand-eye-calibration/data/20251020_125832/T_346522075401.npy')     # 90
T_old = np.load("/home/necl/Projects/hand-eye-calibration/data/20251020_125426/T_346522075401.npy")

print(T_new)
print(T_old)