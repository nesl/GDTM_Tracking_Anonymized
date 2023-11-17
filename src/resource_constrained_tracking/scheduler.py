import numpy as np
from pubsub import Publisher,Subscriber
from simulator import SimulatorObject
from tracker import MultiObsKalmanFilter

class SimpleScheduler(SimulatorObject):
    def __init__(self,id="Sched1",max_sensors=2, dt=0.5, std_acc=2, broker=None, sensors=[]):
        super().__init__(id,dt)
        self.broker=broker
        self.tracker = MultiObsKalmanFilter(0.5,std_acc)
        self.sensors=sensors

        self.constraints = [lambda a: np.sum(a)<=max_sensors]
        self.objective   = self.info_gain_objective

        self.control_publisher=Publisher("control",broker)
        self.detection_subscriber = Subscriber("detections",broker)
        self.track_subscriber = Subscriber("tracks",broker)

        self.tracker.predict()

    def entropy(self,S):
        D = S.shape[0]
        return 0.5*np.log(np.linalg.det(S+ 0.001*np.eye(4))) + D/2*(1+np.log(2*np.pi))
        
    def info_gain_objective(self,a, x_pred, P_pred, sensor_covs_pred, tracker):
        K = np.sum(a)
        obs_pos = np.tile(x_pred,[1,K])
        covs = [c for i,c in enumerate(sensor_covs_pred) if a[i]==1]    
        _, P_new = tracker.update(obs_pos,covs,store=False)
        return(self.entropy(P_pred)-self.entropy(P_new))

    def optimize(self,a,args):
        D = len(a)
        max_val = -np.inf
        for i in range(1,2**D):
            a = np.array([int(x) for x in np.binary_repr(i,D)])
            for c in self.constraints:
                if(c(a)):
                    val = self.objective(a,*args)
                    if(val>max_val):
                        max_val=val
                        best_a = a
                        
        return(best_a,max_val)

    def step(self):

        #Get most recent track location
        #if any track locations have been published
        if(self.track_subscriber.have_msg()):
            msg   = self.track_subscriber.pop(-1)
            id    = msg["id"]
            pos   = msg["pos"]
            color = msg["col"]
            P     =  msg["cov"]
            self.track_subscriber.clear()

            self.tracker.x = pos
            self.tracker.P = P

        #Predict ahead
        self.tracker.predict()
        x_pred = np.array(self.tracker.x)[:2]
        P_pred = self.tracker.P

        z_cov_pred=[]
        for sensor in self.sensors:
            this_cov,_ = sensor.get_cov(x_pred)
            z_cov_pred.append(this_cov)

        K=len(z_cov_pred)
        a,selection_objective = self.optimize(np.zeros((K,)),[x_pred, P_pred,z_cov_pred,self.tracker])
        #self.control_publisher.publish({"id":self.id, "num_samples":K})
        for i in range(K):
            if a[i]==1:
                self.control_publisher.publish({"id":self.id, "sample":self.sensors[i].id})

