import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import matplotlib.axes.Axes as Axes
from matplotlib.patches import ArrowStyle
import sympy as sp
import sympy.vector as spvec
# from sympy import solve
from sympy import lambdify
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

from analytical_ik import IK

class linkage_robot():
    #origin
    World = spvec.CoordSys3D('World')

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

    link_names = ["Upper Arm",
                    "Forearm Gearbox Output",
                    "Forearm Linkage",
                    "Forearm",
                    "Upper Arm Grounded Linkage",
                    "Triangle",
                    "Forearm Grounded Linkage",
                    "End Effector"]
    link_names_simple = link_names[:4]

    link_names_to_link_objects = [["l1"],
                                  ["a1"],
                                  ["a2"],
                                  ["a3","l2"],
                                  ["b1"],
                                  ["t1","t2","t3"],
                                  ["b2"],
                                  ["ee1","ee2","ee3"]]
    

    link_colors = {link:"teal" for link in grounded_links}
    link_colors = link_colors | {link:"darkviolet" for link in simple_links}

    x_window = [0,0]
    y_window = [0,0]

    def __init__(self):
        self.calculate_b1_pos()
        _ = self.calculate_kinematics()
        self.symbolic_forward_kinematics()
        return
    
    def set_joint_position(self, joint_angles):
        self.joint_angles(joint_angles)
    
    def calculate_b1_pos(self,b1_base = None):
        if b1_base is None:
            self.b1_base = np.array([self.t1*np.cos(self.t1_start_angle), self.t1*np.sin(self.t1_start_angle), 0 ]).reshape(-1,1)
        else:
            self.b1_base = b1_base

    def track_plot_window_size_(self, point, inflate = 0.0):
        '''keeps track of the maximum figure window size necessary when adding annotations'''
        try:
            if point[0] < self.x_window[0]:
                self.x_window[0] = point[0] - inflate
            if point[0] > self.x_window[1]:
                self.x_window[1] = point[0] + inflate
            if point[1] < self.y_window[0]:
                self.y_window[0] = point[1] - inflate
            if point[1] > self.y_window[1]:
                self.y_window[1] = point[1] + inflate
        except:
            pass
    
    def forward_kinematics(self,joint_angles, simple = True, full_state = False):
        '''Calcualtes forward kinematics for the arm
        joint_angles'''
        
        return self.fk_simple_(joint_angles[0],joint_angles[1])
    
    def inverse_kinematics(self,pos,simple = True):
        '''Calculates joint angles given the wrist position'''
        try:
            sol1, sol2 = IK(pos[0],pos[1],self.l1,self.l2)
        except ValueError:
            return [np.nan, np.nan]
        
        if sol1[0] >= 0:
            return np.array(sol1)
        else:
            return np.array(sol2)

    def calculate_kinematics(self, joint_angles = None):
        if not joint_angles is None:
            self.joint_angles = joint_angles
        
        #calcualte positons for main arm
        self.link_l1 = np.array([[0,0,0],[self.l1*np.cos(self.joint_angles[0]), self.l1*np.sin(self.joint_angles[0]), 0]]).T
        self.links['l1'] = self.link_l1
        self.link_a1 = np.array([[0,0,0],
                                 [-self.a1*np.cos(-self.joint_angles[1]), 
                                  self.a1*np.sin(-self.joint_angles[1]), 0]]).T
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

    def symbolic_forward_kinematics(self):
        '''returns the symbolic formulas for all links'''
        l1,l2,theta1,theta2, = sp.symbols(('l1','l2','theta1','theta2',))
        a1,a2,a3 = sp.symbols(('a1','a2','a3'))
        b1, b2 = sp.symbols(('b1','b2'))
        t1,t2,t3 = sp.symbols(('t1','t2','t3'))
        e1,e2,e3 = sp.symbols(('e1','e2','e3'))

        #simple forward kinematics
        origin = sp.Matrix([0,0,0])
        l1_vec = l1* sp.Matrix([sp.cos(theta1), 0, sp.sin(theta1)])
        l2_vec = l2* sp.Matrix([-sp.cos(theta2), 0, -sp.sin(theta2)])
        wrist_pos = l1_vec + l2_vec

        self.l1_vec_s = l1_vec
        self.l2_vec_s = l2_vec
        self.l1_s = sp.Matrix.hstack(origin,l1_vec)

        self.fk_simple_ = lambdify([theta1, theta2], wrist_pos)
    
    def plot_link(self, link, ax, color = 'b'):
        ax.plot(link[0,:],link[1,:],color)
    
    def plot_robot(self, ax, simple = False, colors = None):

        ax.axis('equal')
        if colors is None:
            colors = self.link_colors
        else:
            if isinstance(colors,str):
                colors = {key:colors for key in self.link_colors}
            else:
                colors = self.link_colors | colors

        for link in self.simple_links:
            self.plot_link(self.links[link],ax,color = colors[link])

        if not simple:
            for link in self.grounded_links:
                self.plot_link(self.links[link],ax,color = colors[link])
            
        self.update_plot_lims(ax) #does not handle plot resizing well

    def draw_torque(self, ax, link, value, center = 0, color = 'k', scale = 0.05, text = ""):
        '''draws a curved arrow representing torque.'''

        a = (180.0-45)/2
        if value > 0:
            curvature = 1
            arrowstyle = patches.ArrowStyle.CurveB()
            ang = np.pi/2
        else:
            curvature = -1
            arrowstyle = patches.ArrowStyle.CurveB()
            ang = -np.pi/2

        torque_center = link[:2,center]
        link_vector = link[:,-1]-link[:,center]
        link_vector_hat = link_vector/np.linalg.norm(link_vector)
        offset_angle = np.rad2deg(np.arccos(link_vector_hat[0]))

        start_point = torque_center + link_vector_hat[:2]*scale
        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang),  np.cos(ang)]])
        end_point = torque_center + rot@link_vector_hat[:2]*scale
        angleA = curvature*(180-a) + offset_angle
        angleB= curvature*(90-(180-a))  + offset_angle
        connectionstyle = patches.ConnectionStyle.Angle3(angleA=angleA, angleB=angleB)

        a = patches.FancyArrowPatch(start_point, end_point,
                             connectionstyle=connectionstyle, 
                             arrowstyle=arrowstyle,
                             mutation_scale=10)
        ax.add_patch(a)
        text_point = end_point + np.array([0, 0.03])
        if not text == "":
            ax.annotate(text,text_point,annotation_clip=False)

        self.track_plot_window_size_(text_point)
        self.track_plot_window_size_(end_point)

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

        ax.annotate("", end_coords, xytext = start_coords, arrowprops = {'arrowstyle':'->'}, annotation_clip=False)
        self.track_plot_window_size_(start_coords)
        self.track_plot_window_size_(end_coords)
        if not text == "":
            if end_coords[0]-start_coords[0] < 0:
                text_coords = end_coords - np.array([0.03,0.03])
            else:
                text_coords = end_coords
            self.track_plot_window_size_(text_coords)
            ax.annotate(text,text_coords,annotation_clip=False)
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

    def calculate_equations_of_motion(self, simple = False):
        '''sets up the equations of motion'''
        F_g1x,F_g1y,F_g2x,F_g2y,T1,T2 = sp.symbols(('F_g1x','F_g1y','F_g2x','F_g2y','T1','T2'))
        F_l1x,F_l1y,F_l1x_,F_l1y_,F_l2x,F_l2y,F_l2x_,F_l2y_,= sp.symbols(('F_l1x','F_l1y',
                   'F_l1x_','F_l1y_',
                   'F_l2x','F_l2y',
                   'F_l2x_','F_l2y_',))
        F_a1,F_a2,F_a1_,F_a2_,= sp.symbols(('F_a1', 'F_a2','F_a1_', 'F_a2_',))
        l1x,l1y,l2x,l2y = sp.symbols(('l1x','l1y','l2x','l2y'))
        if not simple:
            F_g3,F_g3_,F_b1,F_b1_,F_b2,F_b2_ = sp.symbols(('F_g3','F_g3_','F_b1','F_b1_','F_b2','F_b2_',))
            F_t1x,F_t1y,F_t1x_,F_t1y_,F_t2,F_t2_,F_eex,F_eey, = sp.symbols(('F_t1x','F_t1y',
                                                                            'F_t1x_','F_t1y_',
                                                                            'F_t2','F_t2_',
                                                                            'F_eex','F_eey'))
        else:
            F_g3,F_g3_,F_b1,F_b1_,F_b2,F_b2_ = 0,0,0,0,0,0
            F_t1x,F_t1y,F_t1x_,F_t1y_,F_t2,F_t2_,F_eex,F_eey, = 8*(0,)
        
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

        l1_vector = self.World.i*l1_vec[0] + self.World.j*l1_vec[1]
        l2_vector = self.World.i*l2_vec[0] + self.World.j*l2_vec[1]
        a1_vector = self.World.i*a1_vec[0] + self.World.j*a1_vec[1]
        b1_vector = self.World.i*b1_vec[0] + self.World.j*b1_vec[1]
        b2_vector = self.World.i*b2_vec[0] + self.World.j*b2_vec[1]
        t1_vector = self.World.i*t1_vec[0] + self.World.j*t1_vec[1]
        t2_vector = self.World.i*t2_vec[0] + self.World.j*t2_vec[1]
        b1_vector_hat = b1_vector/sp.sqrt(b1_vector.dot(b1_vector))
        b2_vector_hat = b2_vector/sp.sqrt(b2_vector.dot(b2_vector))
        Fl1_vector = self.World.i*F_l1x + self.World.j*F_l1y
        Ft1_vector = self.World.i*F_t1x + self.World.j*F_t1y
        Ft1_vector_ = self.World.i*F_t1x_ + self.World.j*F_t1y_
        e1_vector = self.World.i*e1_vec[0] + self.World.j*e1_vec[1]
        e2_vector = self.World.i*e2_vec[0] + self.World.j*e2_vec[1]
        
        #linakge forces - constraint is only valid for a static scenario
        F_a1x = F_a1 * a2_hat[0]
        F_a1y = F_a1 * a2_hat[1]
        F_a1x_ = F_a1_ * a2_hat[0]
        F_a1y_ = F_a1_ * a2_hat[1]
        constraint_Fa1 = F_a1 + F_a1_

        F_a2x = F_a2 * a2_hat[0]
        F_a2y = F_a2 * a2_hat[1]
        F_a2x_ = F_a2_ * a2_hat[0]
        F_a2y_ = F_a2_ * a2_hat[1]
        constraint_Fa2 = F_a2 + F_a2_
        

        Fa1_vector = self.World.i*F_a1x + self.World.j*F_a1y
        Fa2_vector = self.World.i*F_a2x + self.World.j*F_a2y
        Fa2_vector_ = self.World.i*F_a2x_ + self.World.j*F_a2y_
        
        #applies implicit constraint that simple links have only axial forces in static configuration
        Fb1_vector = F_b1*b1_vector_hat
        Fb1_vector_ = F_b1_*b1_vector_hat
        F_g3_vector = F_g3*b1_vector_hat
        F_g3x = F_g3_vector.dot(self.World.i)
        F_g3y = F_g3_vector.dot(self.World.j)
        F_b1x = Fb1_vector.dot(self.World.i)
        F_b1x_ = Fb1_vector_.dot(self.World.i)
        F_b1y = Fb1_vector.dot(self.World.j)
        F_b1y_ = Fb1_vector_.dot(self.World.j)
        Fb2_vector = F_b2*b2_vector_hat
        Fb2_vector_ = F_b2_*b2_vector_hat
        Ft2_vector = F_t2*b2_vector_hat
        Ft2_vector_ = F_t2_*b2_vector_hat
        F_t2x = Ft2_vector.dot(self.World.i)
        F_t2y = Ft2_vector.dot(self.World.j)
        F_t2x_ = Ft2_vector_.dot(self.World.i)
        F_t2y_ = Ft2_vector_.dot(self.World.j)
        F_b2x = Fb2_vector.dot(self.World.i)
        F_b2y = Fb2_vector.dot(self.World.j)
        F_b2x_ = Fb2_vector_.dot(self.World.i)
        F_b2y_ = Fb2_vector_.dot(self.World.j)

        #link l1
        force_balance_l1x = F_g1x + F_l1x + F_t1x_
        force_balance_l1y = F_g1y + F_l1y + F_t1y_
        torque_balance_l1 = T1 + spvec.dot(self.World.k,
                                           spvec.cross(l1_vector,Fl1_vector)) + spvec.dot(self.World.k,
                                           spvec.cross(l1_vector,Ft1_vector_))
        constraint_l1x = F_l1x + F_l1x_
        constraint_l1y = F_l1y + F_l1y_
        # print(l1_vector)
        # print(torque_balance_l1)
        #link a1
        force_balance_a1x = F_g2x + F_a1x
        force_balance_a1y = F_g2y + F_a1y
        torque_balance_a1 = T2 + spvec.dot(self.World.k,spvec.cross(a1_vector,Fa1_vector))

        #link b1
        force_balance_b1x = F_g3x + F_b1x
        force_balance_b1y = F_g3y + F_b1y
        constraint_b1 = F_b1 + F_b1_

        #link t1
        force_balance_t1x = F_b1x_ + F_t2x + F_t1x
        force_balance_t1y = F_b1y_ + F_t2y + F_t1y
        # print("t1_vec")
        # print(t1_vector)
        # print("t2_vec")
        # print(t2_vector)
        torque_balance_t1 = spvec.dot(self.World.k,
                                      spvec.cross(t1_vector,Fb1_vector_)) + spvec.dot(self.World.k,
                                      spvec.cross(t2_vector,Ft2_vector))
        constraint_Ft1x = F_t1x + F_t1x_
        constraint_Ft1y = F_t1y + F_t1y_
        constraint_Ft2 = F_t2 + F_t2_

        #link b2
        force_balance_b2x =  F_t2x_ + F_b2x
        force_balance_b2y = F_t2y_ + F_b2y
        constraint_b2 = F_b2_ + F_b2

        #link a2
        #can be simplified out, but included for visualization purposes
        force_balance_a2x = F_a1x_ + F_a2x
        force_balance_a2y = F_a1y_ + F_a2y
        # print(force_balance_a2x)
        #constraint that forces are alligned with link vector applied above

        #link l2
        a3_vec = (self.link_a3[:,1] - self.link_a3[:,0])
        a3_vector = self.World.i*a3_vec[0] + self.World.j*a3_vec[1]
        Fl2_vector = self.World.i*F_l2x + self.World.j*F_l2y
        force_balance_l2x = F_a2x_ + F_l1x_ + F_l2x
        force_balance_l2y = F_a2y_ + F_l1y_ + F_l2y
        torque_balance_l2 = spvec.dot(self.World.k,spvec.cross(a3_vector,Fa2_vector_) + spvec.cross(l2_vector,Fl2_vector))
        constraint_Fl2x = F_l2x + F_l2x_
        constraint_Fl2y = F_l2y + F_l2y_
        # print(torque_balance_l2)

        #link ee
        Fee_vector = F_eex*self.World.i + F_eey*self.World.j
        force_balance_eex = F_l2x_ + F_b2x_ + F_eex
        force_balance_eey = F_l2y_ + F_b2y_ + F_eey
        torque_balance_ee = spvec.dot(self.World.k,
                                      spvec.cross(e1_vector,Fb2_vector_)) + spvec.dot(self.World.k,
                                      spvec.cross(e2_vector,Fee_vector))
        # print(torque_balance_l2)
        equations = [
                    force_balance_l1x,
                    force_balance_l1y,
                    torque_balance_l1,
                    constraint_l1x,
                    constraint_l1y,
                    force_balance_a1x,
                    force_balance_a1y,
                    torque_balance_a1,
                    constraint_Fa1,
                    force_balance_a2x,
                    force_balance_a2y,
                    constraint_Fa2,
                    force_balance_l2x,
                    force_balance_l2y,
                    torque_balance_l2,
                    constraint_Ft1x,
                    constraint_Ft1y,
                    ]
        if not simple:
            equations = equations + [
                                    force_balance_eex,
                                    force_balance_eey,
                                    torque_balance_ee,
                                    force_balance_t1x,
                                    force_balance_t1y,
                                    torque_balance_t1,
                                    constraint_Ft2,
                                    force_balance_b1x,
                                    force_balance_b1y,
                                    constraint_b1,
                                    force_balance_b2x,
                                    force_balance_b2y,
                                    constraint_b2,
                                    constraint_Fl2x,
                                    constraint_Fl2y,
                                    ]
        #solve system of equations
        # print(equations)
        result = sp.solve(equations)


    def calculate_static_loads(self, ee_load = None, torques = None, simple = False):
        '''calculate the max static loads in all links.
        defines a system of equations for the loads in each link assuming 
        the end effector is pinned in place, able to resist any applied load'''

        #symbols
        N = spvec.CoordSys3D('N')
        F_g1x,F_g1y,F_g2x,F_g2y,T1,T2 = sp.symbols(('F_g1x','F_g1y','F_g2x','F_g2y','T1','T2'))
        F_l1x,F_l1y,F_l1x_,F_l1y_,F_l2x,F_l2y,F_l2x_,F_l2y_,= sp.symbols(('F_l1x','F_l1y',
                   'F_l1x_','F_l1y_',
                   'F_l2x','F_l2y',
                   'F_l2x_','F_l2y_',))
        F_a1,F_a2,F_a1_,F_a2_,= sp.symbols(('F_a1', 'F_a2','F_a1_', 'F_a2_',))
        l1x,l1y,l2x,l2y = sp.symbols(('l1x','l1y','l2x','l2y'))
        if not simple:
            F_g3,F_g3_,F_b1,F_b1_,F_b2,F_b2_ = sp.symbols(('F_g3','F_g3_','F_b1','F_b1_','F_b2','F_b2_',))
            F_t1x,F_t1y,F_t1x_,F_t1y_,F_t2,F_t2_,F_eex,F_eey, = sp.symbols(('F_t1x','F_t1y',
                                                                            'F_t1x_','F_t1y_',
                                                                            'F_t2','F_t2_',
                                                                            'F_eex','F_eey'))
        else:
            F_g3,F_g3_,F_b1,F_b1_,F_b2,F_b2_ = 0,0,0,0,0,0
            F_t1x,F_t1y,F_t1x_,F_t1y_,F_t2,F_t2_,F_eex,F_eey, = 8*(0,)
        
        if not torques is None:
            raise NotImplementedError
        
        if ee_load is None:
            raise NotImplementedError
        else:
            if simple:
                F_l2x = ee_load[0]
                F_l2y = ee_load[1]
            else:
                F_eex = ee_load[0]
                F_eey = ee_load[1]

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

        l1_vector = N.i*l1_vec[0] + N.j*l1_vec[1]
        l2_vector = N.i*l2_vec[0] + N.j*l2_vec[1]
        a1_vector = N.i*a1_vec[0] + N.j*a1_vec[1]
        b1_vector = N.i*b1_vec[0] + N.j*b1_vec[1]
        b2_vector = N.i*b2_vec[0] + N.j*b2_vec[1]
        t1_vector = N.i*t1_vec[0] + N.j*t1_vec[1]
        t2_vector = N.i*t2_vec[0] + N.j*t2_vec[1]
        b1_vector_hat = b1_vector/sp.sqrt(b1_vector.dot(b1_vector))
        b2_vector_hat = b2_vector/sp.sqrt(b2_vector.dot(b2_vector))
        Fl1_vector = N.i*F_l1x + N.j*F_l1y
        Ft1_vector = N.i*F_t1x + N.j*F_t1y
        Ft1_vector_ = N.i*F_t1x_ + N.j*F_t1y_
        e1_vector = N.i*e1_vec[0] + N.j*e1_vec[1]
        e2_vector = N.i*e2_vec[0] + N.j*e2_vec[1]
        
        #linakge forces - constraint is only valid for a static scenario
        F_a1x = F_a1 * a2_hat[0]
        F_a1y = F_a1 * a2_hat[1]
        F_a1x_ = F_a1_ * a2_hat[0]
        F_a1y_ = F_a1_ * a2_hat[1]
        constraint_Fa1 = F_a1 + F_a1_

        F_a2x = F_a2 * a2_hat[0]
        F_a2y = F_a2 * a2_hat[1]
        F_a2x_ = F_a2_ * a2_hat[0]
        F_a2y_ = F_a2_ * a2_hat[1]
        constraint_Fa2 = F_a2 + F_a2_
        

        Fa1_vector = N.i*F_a1x + N.j*F_a1y
        Fa2_vector = N.i*F_a2x + N.j*F_a2y
        Fa2_vector_ = N.i*F_a2x_ + N.j*F_a2y_
        # F_t2_vector = N.i*F_t2 + N.j*F_t2y_
        # F_b1x = F_b1 * b1_hat[0]
        # F_b1y = F_b1 * b1_hat[1]
        #applies implicit constraint that simple links have only axial forces in static configuration
        Fb1_vector = F_b1*b1_vector_hat
        Fb1_vector_ = F_b1_*b1_vector_hat
        F_g3_vector = F_g3*b1_vector_hat
        F_g3x = F_g3_vector.dot(N.i)
        F_g3y = F_g3_vector.dot(N.j)
        F_b1x = Fb1_vector.dot(N.i)
        F_b1x_ = Fb1_vector_.dot(N.i)
        F_b1y = Fb1_vector.dot(N.j)
        F_b1y_ = Fb1_vector_.dot(N.j)
        Fb2_vector = F_b2*b2_vector_hat
        Fb2_vector_ = F_b2_*b2_vector_hat
        Ft2_vector = F_t2*b2_vector_hat
        Ft2_vector_ = F_t2_*b2_vector_hat
        F_t2x = Ft2_vector.dot(N.i)
        F_t2y = Ft2_vector.dot(N.j)
        F_t2x_ = Ft2_vector_.dot(N.i)
        F_t2y_ = Ft2_vector_.dot(N.j)
        F_b2x = Fb2_vector.dot(N.i)
        F_b2y = Fb2_vector.dot(N.j)
        F_b2x_ = Fb2_vector_.dot(N.i)
        F_b2y_ = Fb2_vector_.dot(N.j)

        #for plotting
        # self.force_orientations = {}
        self.force_locations = {
                        "F_l1":self.link_l1[:,-1],
                        "F_l1_":self.link_l1[:,-1],
                        "F_g1":self.link_l1[:,0],
                        "F_g2":self.link_a1[:,0],
                        "F_t1":self.link_l1[:,-1],
                        "F_t1_":self.link_l1[:,-1],
                        "T1":self.link_l1[:,0],
                        "T2":self.link_a1[:,0],
                        "F_a1":self.link_a2[:,0],
                        "F_a1_":self.link_a2[:,0],
                        "F_a2":self.link_a2[:,-1],
                        "F_a2_":self.link_a2[:,-1],
                        "F_b1":self.link_b1[:,-1],
                        "F_b1_":self.link_b1[:,-1],
                        "F_t2":self.link_b2[:,0],
                        "F_t2_":self.link_b2[:,0],
                        "F_b2":self.link_b2[:,1],
                        "F_b2_":self.link_b2[:,1],
                        "F_l2":self.link_l2[:,-1],
                        "F_l2_":self.link_l2[:,-1],
                        "F_ee":self.link_ee2[:,-1],
                        "F_g3":self.link_b1[:,0]
                        }

        #link l1
        force_balance_l1x = F_g1x + F_l1x + F_t1x_
        force_balance_l1y = F_g1y + F_l1y + F_t1y_
        torque_balance_l1 = T1 + spvec.dot(N.k,
                                           spvec.cross(l1_vector,Fl1_vector)) + spvec.dot(N.k,
                                           spvec.cross(l1_vector,Ft1_vector_))
        constraint_l1x = F_l1x + F_l1x_
        constraint_l1y = F_l1y + F_l1y_
        # print(l1_vector)
        # print(torque_balance_l1)
        #link a1
        force_balance_a1x = F_g2x + F_a1x
        force_balance_a1y = F_g2y + F_a1y
        torque_balance_a1 = T2 + spvec.dot(N.k,spvec.cross(a1_vector,Fa1_vector))

        #link b1
        force_balance_b1x = F_g3x + F_b1x
        force_balance_b1y = F_g3y + F_b1y
        constraint_b1 = F_b1 + F_b1_

        #link t1
        force_balance_t1x = F_b1x_ + F_t2x + F_t1x
        force_balance_t1y = F_b1y_ + F_t2y + F_t1y
        # print("t1_vec")
        # print(t1_vector)
        # print("t2_vec")
        # print(t2_vector)
        torque_balance_t1 = spvec.dot(N.k,
                                      spvec.cross(t1_vector,Fb1_vector_)) + spvec.dot(N.k,
                                      spvec.cross(t2_vector,Ft2_vector))
        constraint_Ft1x = F_t1x + F_t1x_
        constraint_Ft1y = F_t1y + F_t1y_
        constraint_Ft2 = F_t2 + F_t2_

        #link b2
        force_balance_b2x =  F_t2x_ + F_b2x
        force_balance_b2y = F_t2y_ + F_b2y
        constraint_b2 = F_b2_ + F_b2

        #link a2
        #can be simplified out, but included for visualization purposes
        force_balance_a2x = F_a1x_ + F_a2x
        force_balance_a2y = F_a1y_ + F_a2y
        # print(force_balance_a2x)
        #constraint that forces are alligned with link vector applied above

        #link l2
        a3_vec = (self.link_a3[:,1] - self.link_a3[:,0])
        a3_vector = N.i*a3_vec[0] + N.j*a3_vec[1]
        Fl2_vector = N.i*F_l2x + N.j*F_l2y
        force_balance_l2x = F_a2x_ + F_l1x_ + F_l2x
        force_balance_l2y = F_a2y_ + F_l1y_ + F_l2y
        torque_balance_l2 = spvec.dot(N.k,spvec.cross(a3_vector,Fa2_vector_) + spvec.cross(l2_vector,Fl2_vector))
        constraint_Fl2x = F_l2x + F_l2x_
        constraint_Fl2y = F_l2y + F_l2y_
        # print(torque_balance_l2)

        #link ee
        Fee_vector = F_eex*N.i + F_eey*N.j
        force_balance_eex = F_l2x_ + F_b2x_ + F_eex
        force_balance_eey = F_l2y_ + F_b2y_ + F_eey
        torque_balance_ee = spvec.dot(N.k,
                                      spvec.cross(e1_vector,Fb2_vector_)) + spvec.dot(N.k,
                                      spvec.cross(e2_vector,Fee_vector))
        # print(torque_balance_l2)
        equations = [
                    force_balance_l1x,
                    force_balance_l1y,
                    torque_balance_l1,
                    constraint_l1x,
                    constraint_l1y,
                    force_balance_a1x,
                    force_balance_a1y,
                    torque_balance_a1,
                    constraint_Fa1,
                    force_balance_a2x,
                    force_balance_a2y,
                    constraint_Fa2,
                    force_balance_l2x,
                    force_balance_l2y,
                    torque_balance_l2,
                    constraint_Ft1x,
                    constraint_Ft1y,
                    ]
        if not simple:
            equations = equations + [
                                    force_balance_eex,
                                    force_balance_eey,
                                    torque_balance_ee,
                                    force_balance_t1x,
                                    force_balance_t1y,
                                    torque_balance_t1,
                                    constraint_Ft2,
                                    force_balance_b1x,
                                    force_balance_b1y,
                                    constraint_b1,
                                    force_balance_b2x,
                                    force_balance_b2y,
                                    constraint_b2,
                                    constraint_Fl2x,
                                    constraint_Fl2y,
                                    ]
        #solve system of equations
        # print(equations)
        result = sp.solve(equations)
        # print(result)

        results = {}
        results["F_l1"] = np.array([result[F_l1x],result[F_l1y]], dtype = float)
        results["F_l1_"] = np.array([result[F_l1x_],result[F_l1y_]], dtype = float)
        if simple:
            results["F_l2"] = np.array([float(F_l2x),float(F_l2y)], dtype = float)
            results["F_t1"] = np.array([F_t1x,F_t1y], dtype = float)
            results["F_t1_"] = np.array([F_t1x_,F_t1y_], dtype = float)
        else:
            results["F_l2"] = np.array([result[F_l2x],result[F_l2y]], dtype = float)
            results["F_l2_"] = np.array([result[F_l2x_],result[F_l2y_]], dtype = float)
            results["F_g3"] = np.array(result[F_g3] * b1_hat.reshape(-1)[:2], dtype = float)
            results["F_b1"] = np.array(result[F_b1] * b1_hat.reshape(-1)[:2], dtype = float)
            results["F_b1_"] = np.array(result[F_b1_] * b1_hat.reshape(-1)[:2], dtype = float)
            results["F_b2"] = np.array(result[F_b2] * b2_hat.reshape(-1)[:2], dtype = float)
            results["F_b2_"] = np.array(result[F_b2_] * b2_hat.reshape(-1)[:2], dtype = float)
            results["F_t1"] = np.array([result[F_t1x],result[F_t1y]], dtype = float)
            results["F_t1_"] = np.array([result[F_t1x_],result[F_t1y_]], dtype = float)
            results["F_t2"] = np.array(result[F_t2] * b2_hat.reshape(-1)[:2], dtype = float)
            results["F_t2_"] = np.array(result[F_t2_] * b2_hat.reshape(-1)[:2], dtype = float)
            results["F_ee"] = np.array([float(F_eex),float(F_eey)], dtype = float)
        results["F_g1"] = np.array([result[F_g1x],result[F_g1y]], dtype = float)
        results["F_g2"] = np.array([result[F_g2x],result[F_g2y]], dtype = float)
        results["F_a1"] = np.array(result[F_a1] * a2_hat.reshape(-1)[:2], dtype = float)
        results["F_a1_"] = np.array(result[F_a1_] * a2_hat.reshape(-1)[:2], dtype = float)
        results["F_a2"] = np.array(result[F_a2] * a2_hat.reshape(-1)[:2], dtype = float)
        results["F_a2_"] = np.array(result[F_a2_] * a2_hat.reshape(-1)[:2], dtype = float)
        results["T1"] = float(result[T1])
        results["T2"] = float(result[T2])
        # print(results)
        
        upper_arm_link_to_forces = ["F_l1", "F_g1", "T1","F_t1_"]
        forearm_link_to_forces = ["F_l1_", "F_a2_", "F_l2"]
        a1_link_to_forces = ["T2","F_a1","F_g2"]
        a2_link_to_forces = ["F_a2","F_a1_"]#need to flesh this out
        upper_arm_grounded_link_forces = ["F_g3","F_b1"]
        forearm_grounded_link_forces = ["F_t2_","F_b2"]
        triangle_forces = ["F_t1","F_t2","F_b1_"]
        ee_forces = ["F_ee", "F_b2_", "F_l2_"]

        self.forces_per_link = {
            self.link_names[0]:upper_arm_link_to_forces,
            self.link_names[1]:a1_link_to_forces,
            self.link_names[2]:a2_link_to_forces,
            self.link_names[3]:forearm_link_to_forces,
            self.link_names[4]:upper_arm_grounded_link_forces,
            self.link_names[5]:triangle_forces,
            self.link_names[6]:forearm_grounded_link_forces,
            self.link_names[7]:ee_forces,
        }
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
    # def plot_force(self, ax, pos, vector, scale = 0.01, color='k'):
        # self.draw_arrow(ax, pos,vector,text=f"{F_l2:.1f}N")

    def update_plot_lims(self, ax : plt.axis, inflate = 0.0):
        x_lims = ax.get_xlim()
        y_lims = ax.get_ylim()
        
        if x_lims[0] < self.x_window[0]:
            self.x_window[0] = x_lims[0]
        if x_lims[1] > self.x_window[1]:
            self.x_window[1] = x_lims[1]
        if y_lims[0] < self.y_window[0]:
            self.y_window[0] = y_lims[0]
        if y_lims[1] > self.y_window[1]:
            self.y_window[1] = y_lims[1]

        self.x_window[0] -= inflate
        self.x_window[1] += inflate
        self.y_window[0] -= inflate
        self.y_window[1] += inflate

        ax.set_xlim(self.x_window)
        ax.set_ylim(self.y_window)

    def plot_link_loads(self, results = None, simple = True):
        '''plots loads for each link as a separate free body'''
        if results is None:
            results = self.calculate_static_loads(simple = simple)
        
        if simple:
            nplots = 4
            fig, axes = plt.subplots(2,2)
        else:
            nplots = 8
            fig1, axes1 = plt.subplots(2,2)
            fig2, axes2 = plt.subplots(2,2)
            axes = np.vstack((axes1,axes2))
        
        scale = 0.0005

        for i, ax in enumerate(axes.reshape(-1)):
            if i==nplots:
                continue
            ax.set_title(self.link_names[i])
            self.plot_robot(ax, simple = simple, colors = 'lightgray')
            #highlight link
            for l in self.link_names_to_link_objects[i]:
                self.plot_link(self.links[l],ax,color = "darkblue")

                #add forces
                forces = self.forces_per_link[self.link_names[i]]
                for force in forces:
                    
                    if force[0]=="T":
                        torque = results[force]
                        # torque_location = self.force_locations[force][:2]
                        torque_link = self.links[l]
                        # torque_link = self.links['l1']
                        self.draw_torque(ax, torque_link, torque, 0, text=f"{torque:.1f}Nm")
                        continue

                    scalar_force = np.linalg.norm(results[force])
                    # print(force)
                    # print(self.force_locations[force])
                    # print(results[force] * scale)
                    force_location = self.force_locations[force][:2]
                    scaled_force_vector = results[force] * scale
                    force_arrow_endpoint = force_location + scaled_force_vector
                    if scalar_force < 0.01:
                        continue
                    self.draw_arrow(ax, 
                                    force_location,
                                    force_arrow_endpoint,
                                    text=f"{scalar_force:.1f}N")

            self.update_plot_lims(ax)

    def time_analysis(self, iter = 10):
        import time

        st = time.time()
        for _ in range(iter):
            self.inverse_kinematics([0.1,-0.1])
        dur = time.time()-st
        print(f"ik elapsed time = {dur}")


        st = time.time()
        for _ in range(iter):
            self.calculate_kinematics([0.1,-0.1])
        dur = time.time()-st
        print(f"calculate link pos elapsed time = {dur}")

        st = time.time()
        for _ in range(iter):
            self.calculate_static_loads([1,1])
        dur = time.time()-st
        print(f"calculate loads elapsed time = {dur}")


# joint1_angle = 0
# joint2_angle = -np.pi/2
if __name__ == "__main__":

    robot = linkage_robot()
    robot.calculate_kinematics([np.pi/4,-np.pi/4])
    print(robot.forward_kinematics([np.pi/4,-np.pi/4]))
    print(robot.inverse_kinematics(np.array([0,0])))
    # print(robot.inverse_kinematics(np.array([0.5,0])))
    simple = False
    # robot.calculate_kinematics([np.pi/4,-np.pi/4])
    # robot.calculate_kinematics([0,-np.pi/2])
    results = robot.calculate_static_loads( ee_load = [0,10], simple = simple)
    fig, ax = plt.subplots()
    robot.plot_robot(ax, simple=simple)
    robot.plot_ee_load(ax,results, simple = simple)
    # robot.draw_arrow(ax,(0,0),(0.5,0.5),)
    robot.plot_link_loads(results = results, simple = simple)
    robot.time_analysis()
    plt.show()

