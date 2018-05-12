"""
Created on 4/18/18 for 2018 SIMIODE Student Competition Using Differential 
Equations Modeling
Authors: Olivia Boyd, Emily Rexer, Emily Eckels

This program was created to model falling paper and cardboard given an initial 
y-position and an initial wind speed.

The purpose of this model was to find the minimum y-position and wind speed such 
that a certain percentage of falling paper was separated from falling cardboard. 

Definition: An intersection is any position at which the x-value of the paper 
is less than the x-value of the cardboard, because we are supposing that paper 
will always fall farther in the x direction than cardboard.
"""

import numpy as np
from numpy.random import random as rng
import numpy.random as rand
from scipy.integrate import odeint
from math import exp
import math

'============================Define/Initialize variables============================'

X = [] #A list of v_a0 values at some times t_1, t_2, ... t_f
Y = [] #A list of y0 values at some times t_1, t_2, ... t_f
Z = [] #A list of the number of intersections based on v_a0 and y0
distances = [] #A list of the distances from some point (YPOS0, AIRSPEED)

zeros_p = [] #A list of points where y_p(x(t)) = f(x(t)) intersects the x-axis
zeros_c = [] #A list of points where y_c(x(t)) = f(x(t)) intersects the y-axis

numX = 0 #Initial number of intersections between paper and cardboard

NUMOBJECTS = 10 #Number of objects dropped

#Constants
DENSITY_PAPER = 800 #kg/m^3
DENSITY_CB = 700    #kg/m^3
DENSITY_AIR = 1.225 #kg/m^3
Cd = 1.17           #drag constant

#Initial conditions
XPOS0 = 0 #Intial x position in meters
VX0 = 0   #Initial x velocity in meters/second
VY0 = 0   #Initial y velocity in meters/second

#Quantities to be minimized
AIRSPEED = 0.0 #Initial air speed in meters/second; iterations will start at
             #x = AIRSPEED
YPOS0 = 0.0    #Initial y position in meters/second; iterations will start at 
             #y = YPOS0

#Define bounds for iterations
units_x = 1   #Number of unit squares to search in the x-direction
units_y = 1   #Number of unit squares to search in the y-direction
stepSize = .2 #Step size of iterations - 10*stepSize must divide 10

#Set the range of acceptable intersections in fraction form (NUMOBJECTS*_frac/10)
lowerFrac = 3
upperFrac = 4

#Time information
t0=0.0        #Time in seconds
DT=0.001      #Time interval in seconds
SIMTIME = 750 #Number of seconds used in the simulation
'''========================================================================='''

#v_a is a function of xpos
def v_a(AIRSPEED, xpos):
    v_a = AIRSPEED * exp(-xpos) #v_a is modeled as exponential decay
    return[(v_a)]

#Express d^2(x)/dt^2 as dx/dt in order to use odeint
def XAccel(x, t, mass, area, PvCB, theta):
    xpos, vx = x     #Take the two elements of the list x and separate them
    dxdt = [vx, (0.5*DENSITY_AIR*area*Cd*(1-theta*theta/2)*v_a(AIRSPEED, xpos)[0]*(v_a(AIRSPEED, xpos)[0]-2*vx))/mass]
    return dxdt     #Return the derivatives 

#Express d^2(y)/dt^2 as dy/dt
def YAccel_p(y, t, mass, area, s, PvCB, theta, g = 9.8):
    ypos, vy = y     #Take the two elements of the list y and separate them
    dydt = [vy, ((math.pi*theta*area*DENSITY_AIR*vy*vy/(1+(area*2*theta)/(s*s)))/mass)-g]
    return dydt      #Return the derivative

    
def YAccel_c(y, t, mass,area, s, PvCB, theta, g = 9.8):
    ypos, vy = y
    dydt = [vy, (0.5*DENSITY_AIR*area*Cd*theta*vy*vy)/mass - g]
    return dydt      #Return the derivative

#Set initial values and solve the ODEs for x and y as functions of time
def Initialize():
    x0 = [XPOS0, VX0]
    y0 = [YPOS0, VY0]
    
    PvCB = rand.choice([0,1])          #0 = paper, 1 = cardboard
    theta = ((rng(1)[0]+0.001)*3.14)/9 #theta varies between approximately 
                                       #0 degrees and 10 degrees
    
    if PvCB == 0:                #If paper
        i = (rng(1)[0]+0.1)*.594 #i ranges between 0.0594m and 0.594m
        j = (rng(1)[0]+0.1)*.841 #j ranges between 0.0841m and 0.841m
        area = i*j               #Area of the paper
        s= np.sqrt(i*i+j*j)      #Diagonal of the paper
        height = .000005         #Height of paper is 0.05mm
        mass = DENSITY_PAPER * height * area
        #Numerically integrate XAccel and YAccel to find x(t) and y_p(t)
        x = odeint(XAccel, x0, t, args=(mass, area, PvCB, theta))    
        y_p = odeint(YAccel_p, y0, t, args=(mass, area, s, PvCB, theta))
        return [x, y_p, PvCB]
    
    else:                           #If cardboard
        i = (rng(1)[0]+0.001)*1.219 #i ranges between 0.1219m and 1.219m
        j = (rng(1)[0]+0.001)*.457  #j ranges between 0.0457m and 0.457m
        area = i*j                  #Area of the cardboard
        s= np.sqrt(i*i+j*j)         #Diagonal of the cardboard
        height = ((rng(1)[0]* 0.0045) + 0.0015) #Height of cardboard ranges 
                                                #between 0.0015m and 0.006m
        mass = DENSITY_CB * height * area
        #Numerically integrate XAccle and YAccel to find x(t) and y_c(t)
        x = odeint(XAccel, x0, t, args=(mass, area, PvCB, theta))
        y_c = odeint(YAccel_c, y0, t, args=(mass, area, s, PvCB, theta))
        return [x, y_c, PvCB]

def getZero(x,y): #Find the x-value when the object hits the ground
    lengthpos = np.searchsorted(-1*y[:,0],0)
    return(x[lengthpos][0])
    
#Find number of intersections of paper and cardboard, assuming paper always 
#goes farther than cardboard
def intersections(zeros_p,zeros_c,numX):
    #Iterate through the zeros_p and zeros_c lists and compare the values
    for i in range(0, len(zeros_p)):
        for j in range(0, len(zeros_c)):
            #If any piece of cardboard goes farther than any piece of paper, 
            #define an intersection
            if zeros_p[i]<=zeros_c[j]: 
                numX = numX + 1
    return(numX)
    
def distance(YPOS0, AIRSPEED): #find the distance between some point (YPOS0, AIRSPEED)
                               #and the origin
    return(math.sqrt(YPOS0*YPOS0 + AIRSPEED*AIRSPEED))

#Create the array for time t
t=np.arange(t0, SIMTIME+DT, DT)   

#Number of steps used to generate data points
for a in range(0, int(units_x/stepSize) + 1):     #Search area in the x direction
    for b in range(0, int(units_y/stepSize) + 1): #Search area in the y direction        
        #loop that executes NUMOBJECTS times    
        for x in range(0, NUMOBJECTS):
            x, y, PvCB = Initialize()       
        
            if PvCB == 0:                                  #if paper,plt.plot(x[:, 0], y[:, 0], 'k', label='Y') #black is paper 
                zeros_p.append(getZero(x,y))               #add a zero to check
                                                           #for intersections
            else:                                          #if paper,
                zeros_c.append(getZero(x,y))               #add a zero to check
                                                           #for intersections
            intx = (intersections(zeros_p,zeros_c,numX))   #number of intersections
            
        if((round(YPOS0, 1) != 0) and (round(AIRSPEED, 1) != 0)):    #if the point does not
                                                           #lie on the YPOS0-axis
                                                           #or the AIRSPEED-axis
                                                           #(these axes are asymptotes)
            if(intx >= NUMOBJECTS*lowerFrac/10 and intx<= NUMOBJECTS*upperFrac/10):
                #Ordered triples in the y_0, v_a, and intersections plane only 
                #consider the triples if intx falls within the appropriate bounds
                X.append(YPOS0)
                Y.append(AIRSPEED)
                Z.append(intx)
                distances.append(distance(YPOS0, AIRSPEED))
            
        YPOS0 = YPOS0 + stepSize #increase YPOS0 by the step size
        zeros_p=[]               #add zeros to check for intersections later
        zeros_c=[]               #add zeros to check for intersections later
        
        if(a == 0 and b == 0):              #UI details
            print("Computing", end=" ")
        elif(a > 0 and b <= 3 and b > 0): 
            print(".", end=" ")
        elif(a > 0 and b == 4):
            print("\n")
    YPOS0 = round(abs(YPOS0 - (units_y + stepSize)),1) #round and take the absolute
                                                       #to account for floats
    AIRSPEED = AIRSPEED + stepSize                     #incrase AIRSPEED by step size             
    if(a > 0 and a < int(units_x/stepSize) - 1):       #UI details
        print("Still computing", end=" ")
    elif (b <= 3 and b > 0):
        print(".", end=" ")
    elif (a == int(units_x/stepSize) - 1):
        print("Almost done", end=" ")

if(len(distances) != 0): #find the first index of the minimum distance, if one exists
    minIndex = distances.index(min(distances))   
    print("The minimum conditions are as follows: \n", X[minIndex], "feet\n", Y[minIndex], "meters/sec")
else:
    print("The search area was not large enough to find a minimum set of conditions.", end=" ")
    print("Increase the number of squares searched in either the x- or y-direction.")
