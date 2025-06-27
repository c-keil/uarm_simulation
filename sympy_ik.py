import sympy as sp
# from sympy.printing.pycode import NumPyPrinter
import numpy as np

l1,l2,theta2,xee,yee = sp.symbols(('l1','l2','theta2','xee','yee'))

mag = (xee-l2*sp.cos(theta2))**2 + (yee - l2*sp.sin(theta2))**2
# print(mag) 
expanded = sp.expand(mag)
print(expanded)

alt = l2**2 +xee**2 + yee**2 - 2*l2*(xee*sp.cos(theta2) + yee*sp.sin(theta2)) - l1**2
print(alt.subs((l1,l2,xee,yee,theta2),(0.5,0.5,0.7,0,0.3)))
# print(alt)
# print(sp.trigsimp(alt))
print()
# alt = alt.subs(l2,0.5)
# alt = alt.subs(l1,0.5)

solns = sp.solve(alt,theta2)
soln1 = solns[0]
soln2 = solns[1]
# print(soln1)
# print(soln2)
# s1 = sp.lambdify((xee, yee),soln1)
# s2 = sp.lambdify((xee, yee),soln2)

elbow_pos_x = (xee-l2*sp.cos(theta2)).subs(l2,0.5)
elbow_pos_y = yee-l2*sp.sin(theta2)
theta1 = (sp.acos(elbow_pos_x/l1)).subs((l1,l2),(0.5,0.5))
t1 = sp.lambdify((xee,theta2),theta1)

xe = 0.7
ye = 0
# a1 = s1(xe, ye)
# a2 = s2(xe, ye)
# # print(type(a1))
# print(np.rad2deg(a1),np.rad2deg(a2))

# with open('analytical_ik.py', 'w') as f:
#     f.write('import math\n')
#     print("theta2_1 = ", sp.pycode(soln1), file=f)  
#     print("theta2_2 = ", sp.pycode(soln2), file=f)
#     print("theta1 = ", sp.pycode(theta1), file=f)

print("theta2_1 = ", sp.pycode(soln1))  
print("theta2_2 = ", sp.pycode(soln2))
print("theta1 = ", sp.pycode(theta1))


# print(th1(a1,xe),th1(a2,xe))
# print(t1(xe,a1))
# print(t1(xe,a2))
# Eq = sp.Eq(l1**2 , )
# res = sp.solve(theta2,Eq)
# print(res)