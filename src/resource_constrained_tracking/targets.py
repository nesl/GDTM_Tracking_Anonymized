from simulator import SimulatorObject
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 
import time
from pubsub import Publisher
import threading

class PointTarget(SimulatorObject):
    def __init__(self,id="",pos=None,max_speed=1,p_dir_change=0.1,color="b",axes=[],broker=None,dt=0.1):

        super().__init__(id,dt)

        if(pos is None):
            pos = self.rs.randn(2,1)
        
        self.x=pos
        self.v=self.rs.randn(2,1)/10
        self.a=0*pos        
        self.max_speed = max_speed
        self.p_dir_change=p_dir_change
        self.color=color
        self.axes=axes
        self.thread=None

        self.publisher=Publisher("targets",broker)
        
    def step(self):
        #if(self.rs.rand()<self.p_dir_change):
        #    self.a = (self.rs.rand(2,1)-0.5)/5

        d = np.sqrt(np.sum(self.x**2))
        if(d>8):
            self.a = -0.1*(d-5)*self.x/d
        else:
            if(self.rs.rand()<self.p_dir_change):
                self.a = 0.5*self.a + 0.1*(self.rs.randn(2,1))
        ##    else:
         #       self.a=0*self.a


        self.v = self.v + self.a

        speed = np.sqrt(np.sum(self.v**2))
        if(speed>self.max_speed):
            self.v = self.max_speed*self.v/speed
            
        self.x = self.x + self.v
        


        '''
        if(self.x[0]>0.8*self.env.maxx): 
            self.a[0] = -0.2
        if(self.x[0]<0.8*self.env.minx): 
            self.a[0] = 0.2
        if(self.x[1]>0.8*self.env.maxy): 
            self.a[1] = -0.2
        if(self.x[1]<0.8*self.env.miny): 
            self.a[1] =0.2
        '''
        
        t=np.floor(time.time()*10)/10
        self.publisher.publish({"id":self.id, "pos":self.x, "col":self.color,"t":t})

