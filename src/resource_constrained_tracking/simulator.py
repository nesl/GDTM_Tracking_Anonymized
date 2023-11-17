from abc import ABC, abstractmethod
import threading
import time
import numpy as np
import time

class SimulatorObject(ABC):
    def __init__(self, id, dt=0.1,seed=0):
        self.sim = None
        self.id = id
        self.dt=dt
        self.state=0
        self.seed=seed
        self.rs = np.random.RandomState(self.seed)
    
    def get_time(self):
        return self.sim.get_time()
    
    def set_simulator(self,sim):
        self.sim = sim
        self.env = sim.env

    @abstractmethod
    def step(self):
        pass

    def run(self):
        self.rs = np.random.RandomState(self.seed)
        while(self.state!=0):
            t=time.time()
            self.step()
            time_diff = time.time()-t
            time.sleep(max(0.01,self.dt-time_diff))                
            
        return
        
    def start(self):
        print("Starting %s"%self.id)
        self.state=1
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        print("Stopping %s"%self.id)
        self.state=0


class Simulator():
    def __init__(self,name,env,views=[],dt=0.1):
        self.name=name
        self.objects=[]
        self.env=env
        self.views=views
        self.t=0
        self.dt=dt
        self.start_time = np.floor(time.time()*10)/10

        for view in views:
            view.start_time=self.start_time
    
    def time(self):
        return(np.floor(time.time()*10)/10 - self.start_time())

    def add_object(self, obj):
        self.objects.append(obj)
        obj.set_simulator(self)

    def run(self,T):
        for obj in self.objects:
            obj.start()

        t = time.time()
        while(time.time()-t<T):
            t2=time.time()
            for view in self.views:
                view.update()
            self.t=self.t+1
            time.sleep(max(0.01,self.dt-(time.time()-t2)))

        self.stop()

    def stop(self):
        for obj in self.objects:
            obj.stop()

        