#import all useful modules
import numpy as np
import pandas as pd
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate
import astropy.units as u
import astropy.constants as c
import kepler
pd.set_option('display.max_columns', None)
from scipy.integrate import quad

# Name Variables
pi = np.pi
G = 6.6743e-11  #* (10**(-11)) #gravitational constant
year_sec = 3.154e7 #*10**7
half_year = year_sec / 2
jupiter_mass = 1.89e27 #*10**27
sun_mass = 1.989e30 #*10**30 #solar mass
earth_mass = sun_mass/333000 #kg
jupit_mass = 1#.898 * 10**27
sun_mass_jup = 1047#*jupit_mass

#define all functions needed

def 
semi_amplitude(period,planet_mass,star_mass,eccentricity,inclination):
    '''
    This function takes in the period of the orbiting exoplanet, 
    the mass of both the exoplanet and the star, the eccentricity of 
the orbiting exoplanet, and inclination and returns the semi-amplitude of 
    the exoplanet.
    
    '''
    #Variables
    pi = np.pi
    G = 6.6743e-11
    ##############
    
    one = (((2*pi*G)/period)**(1/3))
    two = ((planet_mass)*(np.sin(inclination))) / ((star_mass +planet_mass)**(2/3))
    three = 1/(np.sqrt(1-(eccentricity**2)))
    
    sem_amp = one*two*three
    return sem_amp

def RadialVelocity1(m_1,q,e,i,a,phi,w,t,eps):
    m_2=m_1*q
   
    
    period = 2*np.pi*np.sqrt((a*c.au)**3/(c.G*(m_1+m_2)*c.M_sun))
    n = 2*np.pi/period.values
    
    mean_anomaly = n*(t)+phi
    eccentric_anomaly, cos_true_anomaly, sin_true_anomaly = kepler.kepler(mean_anomaly, e)
    k = np.sqrt(c.G/(((m_1+m_2)*c.M_sun)*(a*c.au)*(1-e**2)))*(m_2*c.M_sun)*np.sin(np.deg2rad(i))
    b = (np.cos(w) * cos_true_anomaly - np.sin(w) * sin_true_anomaly + e * np.cos(w))
    
    rv = k*b
    #print(rv)
    RV = (rv).values
    #sigma = np.array(np.random.uniform(low=2, high=3,size=1),dtype=float)
    noisy_RV = RV+(eps*np.random.randn(*RV.shape))
    #print(period/86400)
    return noisy_RV,k,b

def f_value(a1,a2):
    psi = np.linspace(0,2*pi,num=100000,endpoint=True)
    funct = (np.cos(psi)) / ((1 - (2*(a1/a2))*(np.cos(psi))+((a1/a2)**2))**(3/2))
                             
    fvalue = lambda psi: funct
    f = np.trapz(funct)
    
    return f 

def f2_value(a1,a2):
    psi = np.linspace(0,2*pi,num=100000,endpoint=True)
    funct = (np.cos(2*psi)) / ((1 - (2*(a1/a2))*(np.cos(psi))+((a1/a2)**2))**(3/2))
                             
    fvalue = lambda psi: funct
    f = np.trapz(funct)
    
    return f   

def f_sci(psi, a1, a2):

    return (np.cos(psi)) / ((1 - (2*(a1/a2))*(np.cos(psi))+((a1/a2)**2))**(3/2))

def f2_sci(psi, a1, a2):

    return (np.cos(2*psi)) / ((1 - (2*(a1/a2))*(np.cos(psi))+((a1/a2)**2))**(3/2))

def a_22(m1,m2,big_m,a1,a2,f):
    T = np.sqrt(4*(pi**2)*((a2*1.496E11)**3)/(G*((big_m*(1.89)*10E27)+(m2*(1.89)*10E27))))
    n_2 = 2*pi/T
    one = (n_2) * (1/(4*pi))
    two = m1 / (big_m + m2)
    three = ((a1/a2)**2) * f
    
    a22 = one*two*three
    return a22

def a_21(m1,m2,big_m,a1,a2,f):
    T = np.sqrt(4*(pi**2)*((a2*1.496E11)**3)/(G*((big_m*(1.89)*10E27)+(m2*(1.89)*10E27))))
    n_2 = 2*pi/T    
    one = (-n_2) * (1/(4*pi))
    two = m1 / (big_m + m2)
    three = (a1/a2) * f
    
    a21 = one*two*three
    return a21

def b_22(m1,m2,big_m,a1,a2,f):
    T = np.sqrt(4*(pi**2)*((a2*1.496E11)**3)/(G*((big_m*(1.89)*10E27)+(m2*(1.89)*10E27))))
    n_2 = 2*pi/T    
    one = (-n_2) * (1/(4*pi))
    two = m1 / (big_m + m2)
    three = ((a1/a2)**2) * f
    
    b22 = one*two*three
    return b22

def b_21(m1,m2,big_m,a1,a2,f):
    T =np.sqrt(4*(pi**2)*((a2*1.496E11)**3)/(G*((big_m*(1.89)*10E27)+(m2*(1.89)*10E27))))
    n_2 = 2*pi/T    
    one = (n_2) * (1/(4*pi))
    two = m1 / (big_m + m2)
    
    three = (a1/a2) * f
    b21 = one*two*three
    return b21

def f_ll(a22,a21,e1ic,e2ic,omega1,omega2,b21,i1ic,i2ic,big_omega1,big_omega2,b22):
    
    one = a21*(e1ic/e2ic)*(np.cos((omega1-omega2)))
    two = b21*(i1ic/i2ic)*(np.cos(big_omega1-big_omega2))
    #print(one)
    #print(two)
    answer = a22 + one - two - b22
    return answer

def f_llMin(a22,a21,e1ic,e2ic,b21,i1ic,i2ic,b22):
    
    one = a21*(e1ic/e2ic)
    two = b21*(i1ic/i2ic)
    #print(one)
    #print(two)
    answer = a22 + one - two - b22
    return answer

def f_llMax(a22,a21,e1ic,e2ic,b21,i1ic,i2ic,b22):
    
    one = a21*(e1ic/e2ic)
    two = b21*(i1ic/i2ic)
    #print(one)
    #print(two)
    answer = a22 - one - two - b22
    return answer

def e3_c(big_m,m1,m2,m3,a2,a3,fll):
    G = 6.6743*10**-11  #gravitational constant m^3 kg^-1 s^-2
    
    one = 15/16
    two =((m3*(1.89)*10E27)*(np.sqrt(G)))/(np.sqrt((big_m*(1.89)*10E27) +(m1*(1.89)*10E27) + (m2*(1.89)*10E27)))
    three = ((a2*1.496E11)**(3/2))/((a3*1.496E11)**3)
    four = np.abs(1/fll)
    
    inside = (one*two*three*four)**(2/3)
   
    e3c = np.sqrt(1 - inside)
    #print(inside)
    return e3c
