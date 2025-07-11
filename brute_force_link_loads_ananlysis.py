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
# zpos = np.linspace(-0.75,0.7,z_grid_size)
zpos = np.linspace(-1,1,z_grid_size)
yy,zz = np.meshgrid(ypos, zpos)
ee_loads = (np.array([0,1]),np.array([1,0]),np.array([1,1]),np.array([-1,1]))
max_torque = 32
max_loads = [np.zeros((z_grid_size,y_grid_size)) for _ in ee_loads]
max_torques = []

links_to_check = [
    "Forearm Linkage",
    "Upper Arm Grounded Linkage",
    "Forearm Grounded Linkage",
]
link_max_loads = {l:0 for l in links_to_check}
link_max_load_joint_configs = {}
link_max_load_force_vector = {}

fig, ax = plt.subplots()

#iterate over sample space
for j, y in enumerate(ypos):
    for i, z in enumerate(zpos):
        #note i,j vs x,y complies with matplotlib convention
        pos = [y,z]
        
        #collision check
        joint_pos = robot.inverse_kinematics(pos)
        if np.any(np.isnan(joint_pos)) or not robot.check_joints(joint_pos):
            for ml in max_loads:
                ml[i,j] = np.nan
            continue
        # if z<-0.5:
        #     print(f"position = {y,z}")
        #     print(f"joints = {joint_pos}")
        #     quit()

        #calcaulte max loads
        for max_load, ee_load in zip(max_loads, ee_loads):
            results = robot.compute_static_loads_symbolic(joint_pos, ee_load = ee_load)
            # results = robot.calculate_static_loads( ee_load = ee_load,simple = simple)
            T1, T2 = np.abs(results["T1"]), np.abs(results["T2"])
            T = T1 if T1 > T2 else T2
            t_factor = (max_torque/T)
            load = ee_load * t_factor
            max_load[i,j] = np.linalg.norm(load)

            #get internal loads
            for link in links_to_check:
                # loads = [robot.forces_per_link[k] for k in robot.forces_per_link[].keys()]
                loads = robot.forces_per_link[link]
                load_name = loads[0]
                # print(link, load_name, loads)
                link_load = np.abs(np.linalg.norm(results[load_name]))*t_factor
                if link_load > link_max_loads[link]:
                    # print(link, load_name, link_load)
                    # _ = [print(k) for k in robot.forces_per_link.keys()]
                    # [print(f"Load {k} = {results[k]*t_factor}") for k in results.keys()]
                    link_max_loads[link] = link_load
                    link_max_load_joint_configs[link] = joint_pos
                    link_max_load_force_vector[link] = ee_load
        
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

#plot max load condition for different links
for link_name in links_to_check:
    fig, ax = plt.subplots()
    # link = robot.links[link_name]
    robot.plot_robot(ax, link_max_load_joint_configs[link_name], simple = simple, colors = 'lightgray')
    
    for l in robot.link_names_to_objects[link_name]:
        robot.plot_link2D(l,link_max_load_joint_configs[link_name],ax,color = "darkblue")
    ax.set_title(f'Link {link_name}\nmax load = {link_max_loads[link_name]} at joint pos {link_max_load_joint_configs[link_name]}')

plt.show()

