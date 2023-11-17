import numpy as np
from pubsub import Publisher,Subscriber
from simulator import SimulatorObject

class TrackingStats(SimulatorObject):
    def __init__(self,id="Perf1", dt=0.1, broker=None):
        super().__init__(id,dt)
        self.broker=broker

        self.target_subscriber = Subscriber("targets",broker)
        self.track_subscriber = Subscriber("tracks",broker)
        self.performance_publisher=Publisher("performance",broker)

        self.last_target=None
        self.last_track=None
        self.last_t=None
        self.n=0
        self.total_mae=0

    def step(self):
        if(self.target_subscriber.have_msg()):
            msg=self.target_subscriber.pop(-1)
            self.last_target = np.array(msg["pos"][:2,0]).flatten()
            self.target_subscriber.clear()
            self.last_t = msg["t"]

            if(self.track_subscriber.have_msg()):
                msg=self.track_subscriber.pop(-1)
                self.last_track = np.array(msg["pos"][:2,0]).flatten()
                self.track_subscriber.clear()

            if(self.last_target is not None and self.last_track is not None):
                self.n=self.n+1
                this_mae = np.mean(np.sum(np.abs(self.last_target - self.last_track)))
                self.total_mae = self.total_mae + this_mae

                msg = {"id":self.id, "cmae":self.total_mae/self.n, 
                       "imae":this_mae, "t":self.last_t}
                self.performance_publisher.publish(msg)
                self.last_msg = msg

    def stop(self):
        print("Stopping %s"%self.id)
        metrics = ["imae","cmae"]
        for i,m in enumerate(metrics):
            print("Final %s performance: %.3f"%(m,self.last_msg[m]))
