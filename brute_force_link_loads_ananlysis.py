import numpy as np
import matplotlib.pyplot as plt
from uarm_kinematics_loads_simulation import linkage_robot

robot = linkage_robot()

simple = False
# pos = [np.pi/4, - np.pi/4]
# robot.calculate_kinematics(pos)

x_grid_size = 4
y_grid_size = 5
# positions = np.linspace(0.1,1,grid_size)
xpos = np.linspace(0.1,1,x_grid_size)
ypos = np.linspace(0.1,1,y_grid_size)
xx,yy = np.meshgrid(xpos, ypos)
ee_loads = (np.array([0,1]),np.array([1,0]),np.array([1,1]),np.array([-1,1]))
max_torque = 32
max_loads = [np.zeros((y_grid_size,x_grid_size)) for _ in ee_loads]
max_torques = []


for j, x in enumerate(xpos):
    for i, y in enumerate(ypos):
        #note i,j vs x,y complies with matplotlib convention
        pos = [x,y]
        
        joint_pos = robot.inverse_kinematics(pos)
        if np.any(np.isnan(joint_pos)):
            for ml in max_loads:
                ml[i,j] = np.nan
            continue
        
        robot.calculate_kinematics(joint_pos)

        for max_load, ee_load in zip(max_loads, ee_loads):
            results = robot.calculate_static_loads( ee_load = ee_load,simple = simple)
            T1, T2 = np.abs(results["T1"]), np.abs(results["T2"])
            T = T1 if T1 > T2 else T2
            t_factor = (max_torque/T)
            load = ee_load * t_factor
            
            max_load[i,j] = np.linalg.norm(load)
            
fig, axes = plt.subplots(2,2)
ims = []
for ax, loads, vect in zip(axes.reshape(-1),max_loads,ee_loads):
    print(xx.shape)
    print(yy.shape)
    print(loads.shape)
    pm= ax.pcolormesh(xx, yy, loads, shading='nearest')
    ims.append(pm)
    plt.colorbar(pm, ax = ax, label='Nm')
    ax.set_title(f"force vector {vect}")
    ax.set_xlabel('x')
    ax.set_xlabel('y')
# fig.colorbar(ims[0], ax=axes[0])
plt.show()
# fig, ax = plt.subplots()
# robot.plot_robot(ax, simple=simple)
# robot.plot_ee_load(ax,results, simple = simple)
# # robot.draw_arrow(ax,(0,0),(0.5,0.5),)
# robot.plot_link_loads(results = results, simple = simple)
# plt.show()
