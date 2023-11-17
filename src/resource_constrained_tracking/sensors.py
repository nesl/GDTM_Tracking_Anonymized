from simulator import SimulatorObject
import numpy as np
import matplotlib.pyplot as plt
import time
from pubsub import Publisher,Subscriber
from numpy import dot, arccos, clip
from numpy.linalg import norm



class OmniSensor(SimulatorObject):
    def __init__(self,pos,std_scale=0.1,std_min=0.05, color="k",id="",broker=None,dt=0.1,latency=0):
        super().__init__(id,dt)
        self.pos = pos.reshape([2,1])
        self.std_scale = std_scale
        self.std_min = std_min
        self.hist_pos=[]
        self.hist_std=[]
        self.color=color
        self.detections=[]
        self.active=True
        self.latency=latency
        self.det_pos=None
        self.det_cov=None

        self.sensor_publisher=Publisher("sensors",broker)
        self.detection_publisher=Publisher("detections",broker)
        self.target_subscriber=Subscriber("targets",broker)
        self.control_subscriber=Subscriber("control",broker)
    
        self.sensor_publisher.publish({"id":id, "type":"omni", "pos":pos,"col":color,"t":time.time()})

    def get_cov(self,x):
        dist = np.sqrt(np.sum((self.pos-x)**2))
        std  = max(self.std_min,self.std_scale*dist)
        cov  = (std**2)*np.eye(2)
        return cov,0
    
    def step(self,wait=True):

        #Wait until controller indicates to send a sample
        get_sample=False
        while(not get_sample and wait==True):
            msg = self.control_subscriber.pop()
            if( msg is not None and "sample" in msg and msg["sample"]==self.id):
                get_sample=True

        #Get most recent target message
        msg = None
        while(self.target_subscriber.have_msg()):
            msg = self.target_subscriber.pop()

        #Process the message
        if(msg is not None):
            pos = msg["pos"]
            cov,_ = self.get_cov(pos)
            det_pos = np.random.multivariate_normal(pos.flatten(),cov).reshape([2,1])
            self.detection_publisher.publish({"id":self.id, "pos":det_pos, "true_pos":pos, "cov": cov,"col":self.color,"t":time.time()},self.latency)
            self.det_pos=det_pos
            self.det_cov=cov


class DirectionalSensor(SimulatorObject):
    def __init__(self,pos,std_scale=0.1,std_min=0.05, color="k",id="",broker=None,dt=0.1,latency=0):
        super().__init__(id,dt)
        self.pos = pos.reshape([2,1])
        self.std_scale = std_scale
        self.std_min = std_min
        self.hist_pos=[]
        self.hist_std=[]
        self.color=color
        self.detections=[]
        self.active=True
        self.latency=latency

        self.sensor_publisher=Publisher("sensors",broker)
        self.detection_publisher=Publisher("detections",broker)
        self.target_subscriber=Subscriber("targets",broker)
        self.control_subscriber=Subscriber("control",broker)
    
        self.sensor_publisher.publish({"id":id, "type":"omni", "pos":pos,"col":color,"t":time.time()})

    def get_cov(self,x):

        dist = np.sqrt(np.sum((self.pos-x)**2))
        std1  = max(self.std_min,self.std_scale*dist)
        std2  = max(self.std_min,std1/10)
        l = np.diag([std1,std2]) 

        x2 = np.array(x).flatten()
        p2 = np.array(self.pos).flatten()
        dir = x2-p2
        dir = dir/norm(dir)

        dir2 = np.array([-dir[1],dir[0]])

        V = np.hstack([dir.reshape((2,1)),dir2.reshape((2,1))])

        cov = V.dot(l).dot(V.T)

        return cov, 0
    
    def step(self):

        #Wait until controller indicates to send a sample
        get_sample=False
        while(not get_sample):
            msg = self.control_subscriber.pop()
            if( msg is not None and "sample" in msg and msg["sample"]==self.id):
                get_sample=True

        #Get most recent target message
        msg = None
        while(self.target_subscriber.have_msg()):
            msg = self.target_subscriber.pop()

        #Process the message
        if(msg is not None):
            pos = msg["pos"]
            cov,cov_sqrt = self.get_cov(pos)
            det_pos = np.random.multivariate_normal(pos.flatten(),cov).reshape([2,1])
            self.detection_publisher.publish({"id":self.id, "pos":det_pos, "true_pos":pos, "cov": cov,"col":self.color,"t":time.time()},self.latency)

