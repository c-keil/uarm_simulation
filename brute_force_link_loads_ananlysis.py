import numpy as np
import matplotlib.pyplot as plt
from uarm_kinematics_loads_simulation import linkage_robot
import time

simple = False
robot = linkage_robot(simple)

# pos = [np.pi/4, - np.pi/4]
# robot.calculate_kinematics(pos)

y_grid_size = 11
z_grid_size = 21
# positions = np.linspace(0.1,1,grid_size)
ypos = np.linspace(0.1,1,y_grid_size)
zpos = np.linspace(-1,1,z_grid_size)
yy,zz = np.meshgrid(ypos, zpos)
ee_loads = (np.array([0,1]),np.array([1,0]),np.array([1,1]),np.array([-1,1]))
max_torque = 32
max_loads = [np.zeros((z_grid_size,y_grid_size)) for _ in ee_loads]
max_torques = []


fig, ax = plt.subplots()

for j, y in enumerate(ypos):
    for i, z in enumerate(zpos):
        #note i,j vs x,y complies with matplotlib convention
        pos = [y,z]
        
        joint_pos = robot.inverse_kinematics(pos)
        if np.any(np.isnan(joint_pos)) or not robot.check_joints(joint_pos):
            for ml in max_loads:
                ml[i,j] = np.nan
            continue
        
        # robot.calculate_kinematics(joint_pos)

        for max_load, ee_load in zip(max_loads, ee_loads):
            results = robot.compute_static_loads_symbolic(joint_pos, ee_load = ee_load)
            # results = robot.calculate_static_loads( ee_load = ee_load,simple = simple)
            T1, T2 = np.abs(results["T1"]), np.abs(results["T2"])
            T = T1 if T1 > T2 else T2
            t_factor = (max_torque/T)
            load = ee_load * t_factor
            
            max_load[i,j] = np.linalg.norm(load)
        
        robot.plot_robot(ax, joint_pos)
        # plt.draw()
        # time.sleep(0.1)
            
fig, axes = plt.subplots(2,2)
ims = []
for ax, loads, vect in zip(axes.reshape(-1),max_loads,ee_loads):
    ax.axis('equal')
    pm = ax.pcolormesh(yy, zz, loads, shading='nearest')
    ims.append(pm)
    plt.colorbar(pm, ax = ax, label='Nm')
    ax.set_title(f"force vector {vect}")
    ax.set_xlabel('y')
    ax.set_ylabel('z')

plt.show()

