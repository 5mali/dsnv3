import random
import string
import pandas as pd
import numpy as np


# from globalvar import *

#Continuous Adaptive Power Manager using default ENO class
class CAPM (object):
    def __init__(self,location='tokyo', year=2010, shuffle=False, trainmode=False):

        #all energy values i.e. BMIN, BMAX, BOPT, HMAX are in mWhr. Assuming one timestep is one hour
        
        self.BMIN = 0.0                #Minimum battery level that is tolerated. Maybe non-zero also
        self.BMAX = 9250.0            #Max Battery Level. May not necessarily be equal to total batter capacity [3.6V x 2500mAh]
        self.BOPT = 0.5 * self.BMAX    #Optimal Battery Level. Assuming 50% of battery is the optimum
        
        self.HMIN = 0      #Minimum energy that can be harvested by the solar panel.
        self.HMAX = None   #Maximum energy that can be harvested by the solar panel. [500mW]
        
        self.DMAX = 500      #Maximum energy that can be consumed by the node in one time step. [~ 3.6V x 135mA]
        self.N_ACTIONS = 10  #No. of different duty cycles possible
        self.DMIN = self.DMAX/self.N_ACTIONS #Minimum energy that can be consumed by the node in one time step. [~ 3.6V x 15mA]
        
        self.binit = None     #battery at the beginning of day
        self.btrack = []      #track the mean battery level for each day
        self.atrack = []      #track the duty cycles for each day
        self.batt = None      #battery variable
        self.enp = None       #enp at end of hr
        self.henergy = None   #harvested energy variable
        self.fcast = None     #forecast variable
        
        self.location = location
        self.year = year
        self.shuffle = shuffle
        self.trainmode = trainmode
        self.eno = None#ENO(self.location, self.year, shuffle=shuffle, day_balance=trainmode) #if trainmode is enable, then days are automatically balanced according to daytype i.e. day_balance= True
        
        self.violation_flag = False

        self.no_of_day_state = 6;
 
    def reset(self,day=0,batt=-1):
        henergy, fcast, day_end, year_end = self.eno.reset(day) #reset the eno environment
        self.violation_flag = False
        if(batt == -1):
            self.batt = self.BOPT
        else:
            self.batt = batt
            
        self.batt = np.clip(self.batt, self.BMIN, self.BMAX)
        self.binit = self.batt
        self.btrack = np.append(self.btrack, self.batt) #track battery levels

        
        self.enp = self.binit - self.batt #enp is calculated
        self.henergy = np.clip(henergy, self.HMIN, self.HMAX) #clip henergy within HMIN and HMAX
        self.fcast = fcast
        
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
#         norm_fcast = self.fcast/(self.no_of_day_state-1)

        c_state = [norm_batt, norm_enp, norm_henergy] #continuous states
        reward = 0
        
        return [c_state, reward, day_end, year_end]
    
    def getstate(self): #query the present state of the system
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
#         norm_fcast = self.fcast/(self.no_of_day_state-1)
        c_state = [norm_batt, norm_enp, norm_henergy] #continuous states

        return c_state

    #reward function
    def rewardfn(self):
        
        #FIRST REWARD AS A FUNCTION OF DRIFT OF BMEAN FROM BOPT i.e. in terms of BDEV = |BMEAN-BOPT|/BMAX
        bmean = np.mean(self.btrack)
        bdev = np.abs(self.BOPT - bmean)/self.BMAX
        # based on the sigmoid function
        # bdev ranges from bdev = (0,0.5) of BMAX
        p1_sharpness = 10
        n1_sharpness = 20
        shift1 = 0.5
        # r1(x) = 0.5 when x = 0.25. 
        # Therefore, shift = 0.5 to make sure that (2*x-shift) evaluates to zero at x = 0.25

        if(bdev<=0.25): 
            r1 = 2*(1-(1 / (1 + np.exp(-p1_sharpness*(2*bdev-shift1)))))-1
        else: 
            r1 = 2*(1-(1 / (1 + np.exp(-n1_sharpness*(2*bdev-shift1)))))-1
        # r1 ranges from -1 to 1
            
        #SECOND REWARD AS A FUNCTION OF ENP AS LONG AS BMAX/4 <= batt <= 3*BMAX/4 i.e. bdev <= 0.25
        if(bdev <=0.25):
            # enp ranges from enp = (0,3) of DMAX
            p2_sharpness = 2
            n2_sharpness = 2
            shift2 = 6    
            # r1(x) = 0.5 when x = 2. 
            # Therefore, shift = 6 to make sure that (3*x-shift) evaluates to zero at x = 2
#             print('Day energy', np.sum(self.eno.senergy[self.eno.day]))
#             print('Node energy', np.sum(self.atrack)*self.DMAX/self.N_ACTIONS)
#             x = np.abs(np.sum(self.eno.senergy[self.eno.day])-np.sum(self.atrack)*self.DMAX/self.N_ACTIONS )/self.DMAX
            x = np.abs(self.enp/self.DMAX)
            if(x<=2): 
                r2 = (1 / (1 + np.exp(p2_sharpness*(3*x-shift2))))
            else: 
                r2 = (1 / (1 + np.exp(n2_sharpness*(3*x-shift2))))
        else:
            r2 = 0 # if mean battery lies outside bdev limits, then enp reward is not considered.
        # r2 ranges from 0 to 1

        #REWARD AS A FUNCTION OF BATTERY VIOLATIONS
        if(self.violation_flag):
            violation_penalty = 2
        else:
            violation_penalty = 0 #penalty for violating battery limits anytime during the day
        
#         print("Reward ", (r1 + r2 - violation_penalty), '\n')
        return (r1 + r2 - violation_penalty)
        
    
    def step(self, action):
        day_end = False
        year_end = False
        reward = 0
       
        action = np.clip(action, 0, self.N_ACTIONS-1) #action values range from (0 to N_ACTIONS-1)
        self.atrack = np.append(self.atrack, action+1) #track duty cycles
        e_consumed = (action+1)*self.DMAX/self.N_ACTIONS   #energy consumed by the node
        
        self.batt += (self.henergy - e_consumed)
        if(self.batt <= self.BMIN or self.batt >= self.BMAX ):
            self.violation_flag = True #penalty for violating battery limits anytime during the day
        self.batt = np.clip(self.batt, self.BMIN, self.BMAX) #clip battery values within permitted level
        self.btrack = np.append(self.btrack, self.batt) #track battery levels

        
        self.enp = self.binit - self.batt
        
        #proceed to the next time step
        self.henergy, self.fcast, day_end, year_end = self.eno.step()
        self.henergy = np.clip(self.henergy, self.HMIN, self.HMAX) #clip henergy within HMIN and HMAX

        if(day_end): #if eno object flags that the day has ended then give reward
            reward = self.rewardfn()
             
            if (self.trainmode): #reset battery to optimal level if limits are exceeded when training
                self.batt = np.random.uniform(0,1)*self.BMAX
            
            self.violation_flag = False
            self.binit = self.batt #this will be the new initial battery level for next day
            self.btrack = [] #clear battery tracker
            self.atrack = [] #clear duty cycle tracker
                    
                
        norm_batt = self.batt/self.BMAX
        norm_enp = self.enp/(self.BMAX/2)
        norm_henergy = self.henergy/self.HMAX
        norm_fcast = self.fcast/5

        c_state = [norm_batt, norm_enp, norm_henergy] #continuous states
        return [c_state, reward, day_end, year_end]