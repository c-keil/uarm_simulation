import numpy as np

cf_bending_modulus = 100E9
# cf_bending_modulus = 100E9
# cf_bending_modulus = 110E9
# cf_bending_modulus = 85E9

cf_ultimate_strength_compression = 1200E6
# source https://performance-composites.com/carbonfibre/mechanicalproperties_2.asp

expected_load = 1000 #N
# expected_load = 3000 #N

def moment_of_inertia(d1,d2):
    return np.abs(np.pi * (d1**4 - d2**4) / 64)

'''https://www.rockwestcomposites.com/45162-hm.html'''

#tube_dimensions
in2m = 0.0254
outer_diam = 12E-3#0.394*in2m #10E-3 #8.4E-3 #14E-3 #22E-3
inner_diam = 10E-3#(5/16)*in2m #8E-3 #6.35E-3 #12E-3 #20E-3
length = 0.5 #m

cross_section_area = np.pi * ((outer_diam/2)**2 - (inner_diam/2)**2)
I = moment_of_inertia(outer_diam, inner_diam)
r = np.sqrt(I/cross_section_area)

expected_stress = expected_load/cross_section_area
critical_stress = (np.pi**2 * cf_bending_modulus) / (length / r)**2
print(f"max compressive stress: {expected_stress * 1E-6} MPa" )
print(f"critical stress: {critical_stress*1E-6} MPa")
print(f"cf unidirectional ultimate compressive strength: {cf_ultimate_strength_compression*1E-6} MPa")
print(f"tube volume = {length*np.pi*(outer_diam**2 - inner_diam**2)} m^3")