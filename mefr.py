import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *

Ic_pullbar = 236208 # kg * mm^2
m_pullbar = 2.764 # kg
Ic_bracket = 128.944
m_bracket = 0.095
Ic_handle = 91.161
m_handle = 0.237

pos_pivot = np.array([84.24, 78.488]) # mm
pos_pullbar = np.array([451.203, 384.29])
pos_bracket = np.array([396.821, 359.453])
pos_handle = np.array([797.341, 672.739])

d_pullbar = norm(pos_pullbar - pos_pivot) # mm
d_bracket = norm(pos_bracket - pos_pivot)
d_handle = norm(pos_handle - pos_pivot)

I = Ic_pullbar + m_pullbar*(d_pullbar**2) + (Ic_bracket + m_bracket*(d_bracket**2))*2 + Ic_handle + m_handle*(d_handle**2) # kg * mm^2
I = I / (1000*1000) # kg*m^2
print("I: ", I)
t_rep = 1
theta1 = 1.9062
theta0 = 0.8881
angular_acc = (theta1 - theta0)*2*((t_rep/2)**2)
print("alpha: ", angular_acc)
torque = I*angular_acc # N*m
print("tau: ", torque)
force = torque/(d_handle/1000) # N*m / m

print("force: ", force)
print("d_pullbar: ", d_pullbar)
print("d_bracket: ", d_bracket)
print("d_handle : ", d_handle )
print("I_pullbar: ", Ic_pullbar + m_pullbar*(d_pullbar**2), "I_brackets: ", (Ic_bracket + m_bracket*(d_bracket**2))*2, "I_handle: ", Ic_handle + m_handle*(d_handle**2))