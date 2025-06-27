import math

def theta_2(xee, yee, l1, l2):
    theta2_1 =  2*math.atan((2*l2*yee - math.sqrt((-l1**2 + 2*l1*l2 - l2**2 + xee**2 + yee**2)*(l1**2 + 2*l1*l2 + l2**2 - xee**2 - yee**2)))/(-l1**2 + l2**2 + 2*l2*xee + xee**2 + yee**2))
    theta2_2 =  2*math.atan((2*l2*yee + math.sqrt((-l1**2 + 2*l1*l2 - l2**2 + xee**2 + yee**2)*(l1**2 + 2*l1*l2 + l2**2 - xee**2 - yee**2)))/(-l1**2 + l2**2 + 2*l2*xee + xee**2 + yee**2))
    return theta2_1, theta2_2

def theta_1(xee, yee, theta2, l1, l2):
    theta1 =  math.acos((xee - 0.5*math.cos(theta2))/l1)
    return theta1

def IK(xee, yee, l1, l2):
    theta2_1, theta2_2 = theta_2(xee, yee)
    theta1_1 = theta_1(xee, yee, theta2_1, l1, l2)
    theta1_2 = theta_1(xee, yee, theta2_2, l1, l2)
    return [theta1_1, theta2_1], [theta1_2, theta2_2]
    
