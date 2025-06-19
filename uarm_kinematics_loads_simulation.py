import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.axes.Axes as Axes
from matplotlib.patches import ArrowStyle
import sympy as sp
import sympy.vector as spvec
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

class linkage_robot():
    #define robot parameters
    l1 = 0.5
    l2 = 0.5

    a1 = 0.12
    a2 = 0.5
    a3 = 0.12

    b1 = l1
    b2 = l2
    
    #triangle_params
    #computes trainagle and b1 grounded position assuming t1 and t2 are at 
    #90deg separation, and starts with 45 deg separation relative to the main arm zero config
    b1_base = None
    t1 = 0.13
    t2 = 0.1
    t_12_angle = np.radians(-90)
    t1_start_angle = np.radians(135)

    #ee_link
    e1 = t2
    e_12_angle = np.radians(-45)
    e2 = e1*np.cos(e_12_angle)

    joint1_start_angle = 0
    joint2_start_angle = 0
    link_l1_start_vector = np.array([1,0,0]).reshape(3,1)
    link_a2_start_vector = np.array([-1,0,0]).reshape(3,1)

    joint_angles = [joint1_start_angle, joint2_start_angle]

    joint_max_torques = [32,32] #Nm
    
    #Main arm links
    link_l1 = None
    link_a1 = None
    link_a3 = None
    link_a2 = None
    link_l2 = None
    #grounded links
    link_b1 = None
    link_b2 = None
    link_ee1 = None
    link_ee2 = None
    link_ee3 = None
    link_t1 = None
    link_t2 = None
    link_t3 = None

    simple_links = ["l1",
            "a1",
            "a3",
            "a2",
            "l2",]
            
    grounded_links = ["b1", 
            "b2", 
            "ee1", 
            "ee2", 
            "ee3", 
            "t1", 
            "t2", 
            "t3"]
    links = {
        "l1":link_l1,
        "a1" :link_a1,
        "a3" :link_a3,
        "a2" :link_a2,
        "l2" :link_l2,
        "b1" :link_b1,
        "b2":link_b2,
        "ee1":link_ee1,
        "ee2":link_ee2,
        "ee3":link_ee3,
        "t1":link_t1,
        "t2":link_t2,
        "t3":link_t3,
    }

    link_colors = {link:"teal" for link in grounded_links}
    link_colors = link_colors | {link:"darkviolet" for link in simple_links}

    def __init__(self):
        self.calculate_b1_pos()
        _ = self.calculate_kinematics()
        return
    
    def set_joint_position(self, joint_angles):
        self.joint_angles(joint_angles)
    
    def calculate_b1_pos(self,b1_base = None):
        if b1_base is None:
            self.b1_base = np.array([self.t1*np.cos(self.t1_start_angle), self.t1*np.sin(self.t1_start_angle), 0 ]).reshape(-1,1)
        else:
            self.b1_base = b1_base

    def calculate_kinematics(self, joint_angles = None):
        if not joint_angles is None:
            self.joint_angles = joint_angles
        
        #calcualte positons for main arm
        self.link_l1 = np.array([[0,0,0],[self.l1*np.cos(self.joint_angles[0]), self.l1*np.sin(self.joint_angles[0]), 0]]).T
        self.links['l1'] = self.link_l1
        self.link_a1 = np.array([[0,0,0],[-self.a1*np.cos(-self.joint_angles[1]), self.a1*np.sin(-self.joint_angles[1]), 0]]).T
        self.links['a1'] = self.link_a1
    
        self.link_a3 = self.link_a1 + self.link_l1[:,1].reshape(-1,1)
        self.links['a3'] = self.link_a3
        self.link_a2 = np.hstack((self.link_a1[:,1].reshape(-1,1),self.link_a3[:,1].reshape(-1,1)))
        self.links['a2'] = self.link_a2
        link_a3_vec = self.link_a3[:,1]-self.link_a3[:,0]
        link_a3_hat = (link_a3_vec/np.linalg.norm(link_a3_vec)).reshape(-1,1)
        self.link_l2 = np.hstack((self.link_l1[:,1].reshape(-1,1), self.link_l1[:,1].reshape(-1,1)-link_a3_hat*self.l1))
        self.links['l2'] = self.link_l2

        # calcualte grounded linkage positions
        self.link_b1 = self.link_l1 + self.b1_base.reshape(-1,1)
        self.links['b1'] = self.link_b1
        self.link_t1 = np.hstack((self.link_l1[:,-1].reshape(-1,1), self.link_b1[:,-1].reshape(-1,1)))
        self.links['t1'] = self.link_t1
        t1_vec = self.link_t1[:,-1] - self.link_t1[:,0]
        t1_hat = t1_vec/np.linalg.norm(t1_vec)
        
        link_t2_end = self.t2 * (R.from_euler('z', self.t_12_angle).as_matrix() @ t1_hat.reshape(-1,1)) + self.link_l1[:,-1].reshape(-1,1)
        self.link_t2 = np.hstack((self.link_t1[:,0].reshape(-1,1), link_t2_end))
        self.links['t2'] = self.link_t2
        self.link_t3 = np.hstack((self.link_t1[:,-1].reshape(-1,1), self.link_t2[:,-1].reshape(-1,1)))
        self.links['t3'] = self.link_t3

        l2_vec = (self.link_l2[:,-1] - self.link_l2[:,0]).reshape(-1,1)
        self.link_b2 = np.hstack((self.link_t2[:,-1].reshape(-1,1), self.link_t2[:,-1].reshape(-1,1) + l2_vec))
        self.links['b2'] = self.link_b2
        self.link_ee1 = np.hstack((self.link_l2[:,-1].reshape(-1,1), self.link_b2[:,-1].reshape(-1,1)))
        self.links['ee1'] = self.link_ee1
        e1_vec = (self.link_ee1[:,-1] - self.link_ee1[:,0]).reshape(-1,1)
        e1_hat = e1_vec/np.linalg.norm(e1_vec)
        e2_vec = self.e2 * (R.from_euler('z',self.e_12_angle).as_matrix() @ e1_hat)
        self.link_ee2 = np.hstack((self.link_l2[:,-1].reshape(-1,1), self.link_l2[:,-1].reshape(-1,1) + e2_vec))
        self.links['ee2'] = self.link_ee2
        self.link_ee3 = np.hstack((self.link_ee2[:,-1].reshape(-1,1), self.link_ee1[:,-1].reshape(-1,1)))
        self.links['ee3'] = self.link_ee3
        return self.link_l2[:,1].reshape(-1,1)

    def plot_link(self, link, ax, color = 'b'):
        ax.plot(link[0,:],link[1,:],color)
        
    def plot_robot(self, simple = False, colors = None):
        fig, ax = plt.subplots()
        ax.axis('equal')
        # for key in links.keys():
        #     plot_link
        if colors is None:
            colors = self.link_colors
        else:
            colors = self.link_colors | colors
        # print(colors)
        # print(self.link_colors)
        for link in self.simple_links:
            # print(link)
            # print(self.links)
            self.plot_link(self.links[link],ax,color = colors[link])
            # self.plot_link(self.link_l2,ax,color = 'k-')
            # self.plot_link(self.link_a1,ax,color = 'c-')
            # self.plot_link(self.link_a2,ax,color = 'g--')
            # self.plot_link(self.link_a3,ax,color = 'k--')
        # ax.plot(self.link_l2[0,1],self.link_l2[1,1],'rx')

        if not simple:
            for link in self.grounded_links:
                self.plot_link(self.links[link],ax,color = colors[link])
            # self.plot_link(self.link_b1,ax,color = 'r--')
            # self.plot_link(self.link_t1,ax,color = 'y-')
            # self.plot_link(self.link_t2,ax,color = 'y-')
            # self.plot_link(self.link_t3,ax,color = 'y-')
            # self.plot_link(self.link_b2,ax,color = 'r--')
            # self.plot_link(self.link_ee1,ax,color = 'y-')
            # self.plot_link(self.link_ee2,ax,color = 'y-')
            # self.plot_link(self.link_ee3,ax,color = 'y-')
            
        return fig, ax
    
    # def plot_robot_(self):
    #     fig, ax = plt.subplots()
    #     ax.axis('equal')
    #     # for key in links.keys():
    #     #     plot_link
    #     self.plot_link(self.link_l1,ax,color = 'b-')
    #     self.plot_link(self.link_l2,ax,color = 'k-')
    #     self.plot_link(self.link_a1,ax,color = 'c--')
    #     self.plot_link(self.link_a2,ax,color = 'g--')
    #     self.plot_link(self.link_a3,ax,color = 'k--')
    #     ax.plot(self.link_l2[0,1],self.link_l2[1,1],'rx')
    #     return fig, ax
    
    def draw_arrow(self, ax, start, end = None, vector = None, text = ''):
        if end is None and vector is None:
            raise NotImplementedError
        
        start = deepcopy(start)
        end = deepcopy(end)
        vector = deepcopy(vector)

        start_coords = np.array(start).reshape(-1)[:2]
        
        if not end is None:
            end_coords = np.array(end).reshape(-1)[:2]
        else:
            end_coords = start_coords + np.array(vector).reshape(-1)[:2]

        ax.annotate("", end_coords, xytext = start_coords, arrowprops = {'arrowstyle':'->'})
        if not text == "":
            ax.annotate(text,end_coords)
        # ax.annotate('', end_coords, xytext = start_coords, arrowprops = {'width':0.5, 'headwidth':5, 'headlength':5})
    
    def draw_link_force(self,ax,link,value):
        #midpoint
        link_vector = link[:,-1]-link[:,0]
        link_len = np.linalg.norm(link_vector)
        point_a = link[:2,0]
        point_b = (link_vector*0.25)[:2] + point_a
        point_c = (link_vector*0.75)[:2] + point_a
        point_d = link[:2,1]
        midpoint = (link_vector*0.4)[:2] + point_a
        
        if value > 0:
            self.draw_arrow(ax,point_b,end=point_a)
            self.draw_arrow(ax,point_c,end=point_d,)
        else:
            self.draw_arrow(ax,point_a,end=point_b)
            self.draw_arrow(ax,point_d,end=point_c)
        ax.annotate(f"{value:.1f}N",midpoint)

    def plot_link_loads(self, results = None):
        pass

    def calculate_static_loads(self, simple = False):
        '''calculate the max static loads in all links.
        defines a system of equations for the loads in each link assuming 
        the end effector is pinned in place, able to resist any applied load'''

        #symbols
        N = spvec.CoordSys3D('N')
        F_g1x,F_g1y,F_g2x,F_g2y,F_a2,F_l1x,F_l1y,F_l2x,F_l2y,T1,T2 = sp.symbols(('F_g1x','F_g1y',
                   'F_g2x','F_g2y',
                   'F_a2',
                   'F_l1x','F_l1y',
                   'F_l2x','F_l2y',
                    'T1','T2'))
        l1x,l1y,l2x,l2y = sp.symbols(('l1x','l1y','l2x','l2y'))
        if not simple:
            F_b1,F_t1x,F_t1y,F_b2,F_eex,F_eey = sp.symbols(('F_b1',
                                                            'F_t1x','F_t1y',
                                                            'F_b2',
                                                            'F_eex','F_eey'))
        else:
            F_b1,F_t1x,F_t1y,F_b2,F_eex,F_eey = 0,0,0,0,0,0
            
        
        # T1, T2 = 32, 32
        if simple:
            F_l2x = 0
            F_l2y = 50
        else:
            F_eex = 0
            F_eey = 50

        #vectors
        l1_vec = (self.link_l1[:,1] - self.link_l1[:,0])
        l2_vec = (self.link_l2[:,1] - self.link_l2[:,0])
        a1_vec = (self.link_a1[:,1] - self.link_a1[:,0])
        a2_vec = (self.link_a2[:,1] - self.link_a2[:,0])
        a2_hat = a2_vec / np.linalg.norm(a2_vec)
        
        b1_vec = self.link_b1[:,1] - self.link_b1[:,0]
        b2_vec = self.link_b2[:,1] - self.link_b2[:,0]
        b1_hat = b1_vec/np.linalg.norm(b1_vec)
        b2_hat = b2_vec/np.linalg.norm(b2_vec)
        e1_vec = self.link_ee1[:,1] - self.link_ee1[:,0]
        e2_vec = self.link_ee2[:,1] - self.link_ee2[:,0]
        t1_vec = self.link_t1[:,1] - self.link_t1[:,0]
        t2_vec = self.link_t2[:,1] - self.link_t2[:,0]
        # e1_hat = e1_vec/np.linalg.norm(e1_vec)

        l1_vector = N.i*l1_vec[0] + N.j*l1_vec[1] + N.k*0
        l2_vector = N.i*l2_vec[0] + N.j*l2_vec[1] + N.k*0
        a1_vector = N.i*a1_vec[0] + N.j*a1_vec[1]
        b1_vector = N.i*b1_vec[0] + N.j*b1_vec[1]
        b2_vector = N.i*b2_vec[0] + N.j*b2_vec[1]
        t1_vector = N.i*t1_vec[0] + N.j*t1_vec[1]
        t2_vector = N.i*t2_vec[0] + N.j*t2_vec[1]
        b1_vector_hat = b1_vector/sp.sqrt(b1_vector.dot(b1_vector))
        b2_vector_hat = b2_vector/sp.sqrt(b2_vector.dot(b2_vector))
        Fl1_vector = N.i*F_l1x + N.j*F_l1y
        Ft1_vector = N.i*F_t1x + N.j*F_t1y
        e1_vector = N.i*e1_vec[0] + N.j*e1_vec[1]
        e2_vector = N.i*e2_vec[0] + N.j*e2_vec[1]
        
        #linakge forces
        F_a2x = F_a2 * a2_hat[0]
        F_a2y = F_a2 * a2_hat[1]
        Fa2_vector = N.i*F_a2x + N.j*F_a2y
        # F_b1x = F_b1 * b1_hat[0]
        # F_b1y = F_b1 * b1_hat[1]
        Fb1_vector = F_b1*b1_vector_hat
        F_b1x = Fb1_vector.dot(N.i)
        F_b1y = Fb1_vector.dot(N.j)
        Fb2_vector = F_b2*b2_vector_hat
        F_b2x = Fb2_vector.dot(N.i)
        F_b2y = Fb2_vector.dot(N.j)

        #link l1
        force_balance_l1x = F_g1x + F_l1x + F_t1x
        force_balance_l1y = F_g1y + F_l1y + F_t1y
        torque_balance_l1 = T1 + spvec.dot(N.k,
                                           spvec.cross(l1_vector,Fl1_vector)) + spvec.dot(N.k,
                                           spvec.cross(l1_vector,Ft1_vector))

        #link a1
        force_balance_a1x = F_g2x + F_a2x
        force_balance_a1y = F_g2y + F_a2y
        torque_balance_a1 = T2 + spvec.dot(N.k,spvec.cross(a1_vector,Fa2_vector))
    
        #link t1
        force_balance_t1x = -F_b1x + F_b2x + F_t1x
        force_balance_t1y = -F_b1y + F_b2y + F_t1y
        torque_balance_t1 = spvec.dot(N.k,
                                      spvec.cross(t1_vector,Fb1_vector)) + spvec.dot(N.k,
                                      spvec.cross(t2_vector,Fb2_vector))                                           

        #link l2
        a3_vec = (self.link_a3[:,1] - self.link_a3[:,0])
        a3_vector = N.i*a3_vec[0] + N.j*a3_vec[1]
        Fl2_vector = N.i*F_l2x + N.j*F_l2y
        force_balance_l2x = -F_a2x + -F_l1x + F_l2x
        force_balance_l2y = -F_a2y + -F_l1y + F_l2y
        torque_balance_l2 = spvec.dot(N.k,spvec.cross(a3_vector,-Fa2_vector) + spvec.cross(l2_vector,Fl2_vector))

        #link ee
        Fee_vector = F_eex*N.i + F_eey*N.j
        force_balance_eex = -F_l2x + -F_b2x + F_eex
        force_balance_eey = -F_l2y + -F_b2y + F_eey
        torque_balance_ee = spvec.dot(N.k,
                                      spvec.cross(e1_vector,Fb2_vector)) + spvec.dot(N.k,
                                      spvec.cross(e2_vector,Fee_vector))
        # print(torque_balance_l2)
        equations = [force_balance_l1x,
                     force_balance_l1y,
                     torque_balance_l1,
                     force_balance_a1x,
                     force_balance_a1y,
                     torque_balance_a1,
                     force_balance_l2x,
                     force_balance_l2y,
                     torque_balance_l2,
                     ]
        if not simple:
            equations = equations + [force_balance_eex,
                                    force_balance_eey,
                                    torque_balance_ee,
                                    force_balance_t1x,
                                    force_balance_t1y,
                                    torque_balance_t1,]
        #solve system of equations
        result = sp.solve(equations)
        print(result)
        results = {}
        results["F_l1"] = np.array([result[F_l1x],result[F_l1y]], dtype = float)
        if simple:
            results["F_l2"] = np.array([float(F_l2x),float(F_l2y)], dtype = float)
        else:
            results["F_l2"] = np.array([result[F_l2x],result[F_l2y]], dtype = float)
            results["F_b1"] = np.array(result[F_b1] * b1_hat.reshape(-1)[:2], dtype = float)
            results["F_b2"] = np.array(result[F_b2] * b2_hat.reshape(-1)[:2], dtype = float)
            results["F_t1"] = np.array([result[F_t1x],result[F_t1y]], dtype = float)
            results["F_ee"] = np.array([float(F_eex),float(F_eey)], dtype = float)
        results["F_g1"] = np.array([result[F_g1x],result[F_g1y]], dtype = float)
        results["F_g2"] = np.array([result[F_g2x],result[F_g2y]], dtype = float)
        results["F_a2"] = np.array(result[F_a2] * a2_hat.reshape(-1)[:2], dtype = float)
        results["T1"] = float(result[T1])
        results["T2"] = float(result[T2])

        # print("Torque Balance")
        # print(f"F_l1x = {result[F_l1x]}")
        # print(f"F_l1y = {result[F_l1y]}")
        # print(torque_balance_l1)
        # tb = torque_balance_l1.subs(F_l1x,result[F_l1x])
        # tb = tb.subs(F_l1y,result[F_l1y])
        # # tb = tb.subs(F_t1x,result[F_t1x])
        # # tb = tb.subs(T1))
        # print(tb)

        return results
        
    def plot_ee_load(self, ax, results = None, simple = True):
        '''plot force exerted by ee'''
        if results is None:
            results = self.calculate_static_loads()
        # print(results)

        if simple:
            #l2
            F_l2_vec = results["F_l2"]
            F_l2 = np.linalg.norm(F_l2_vec)
            F_l2_vec = F_l2_vec/500
            F_l2_pos = self.link_l2[:2,-1]
            self.draw_arrow(ax, F_l2_pos,vector=-F_l2_vec,text=f"{F_l2:.1f}N")
        else:
            F_ee_vec = results["F_ee"]
            F_ee = np.linalg.norm(F_ee_vec)
            F_ee_vec = F_ee_vec/500
            F_ee_pos = self.link_ee2[:2,-1]
            self.draw_arrow(ax, F_ee_pos,vector=-F_ee_vec,text=f"{F_ee:.1f}N")

        # F_g1_vec = results["F_g1"]
        # F_g2_vec = results["F_g2"]
        # F_g1 = np.linalg.norm(F_g1_vec)
        # F_g1_vec = F_g1_vec/(4*F_g1)
        # F_g1_pos = np.array([0,0])
        # F_g2 = np.linalg.norm(F_g2_vec)
        # F_g2_vec = F_g2_vec/(4*F_g2)
        # F_g2_pos = np.array([0,0])

        # F_a2_vec = results["F_a2"]
        # F_a2 = np.linalg.norm(F_a2_vec)

        # ax.set_xlim([-0.3,1.2])
        # # ax.set_ylim([-0.3,1])
        # self.draw_arrow(ax, F_g1_pos, vector = F_g1_vec)
        # self.draw_arrow(ax, F_g2_pos, vector = F_g2_vec)
        ax.annotate(f"T1: {results["T1"]:.1f} Nm",(-0.1,0.4))
        ax.annotate(f"T2: {results["T2"]:.1f} Nm",(-0.1,0.35))
        # ax.annotate(f"Fg1: {F_g1:.1f}N",(-0.1,0.3))
        # ax.annotate(f"Fg2: {F_g2:.1f}N",(-0.1,0.25))
        # ax.annotate(f"Fa2: {F_a2:.1f} Nm",(-0.1,0.2))

        # self.draw_link_force(ax,self.link_a2,-F_a2)
    def plot_link_loads(self, results = None, simple = True):
        '''plots loads for each link as a separate free body'''
        if results is None:
            results = self.calculate_static_loads(simple = simple)

        
        


# joint1_angle = 0
# joint2_angle = -np.pi/2

robot = linkage_robot()
robot.calculate_kinematics([np.pi/4,0.0])
# robot.calculate_kinematics([np.pi/4,-np.pi/4])
# robot.calculate_kinematics([0,-np.pi/2])
results = robot.calculate_static_loads( simple = True)
fig, ax = robot.plot_robot(simple=True)
robot.plot_ee_load(ax,results, simple = True)
# robot.draw_arrow(ax,(0,0),(0.5,0.5),)
plt.show()