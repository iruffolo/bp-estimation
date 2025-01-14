import numpy as np
from filterpy.kalman import KalmanFilter

d = np.linspace(0, 20, 20)

f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)

f.P *= 1000.0

f.x = np.array([d[0]])
f.F = np.array(
    [
        [
            1.0,
        ]
    ]
)
f.H = np.array([[1.0]])
f.R = np.array([[0.01]])

f.B = np.array([1.0])

for i in range(20):
    f.predict(u=0.01)
    m = d[i]
    f.update(m)
    print(f.x)
    print(f.y)
