import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 
import time
import torch

from pubsub import Publisher,Subscriber
from simulator import SimulatorObject
#from sortedcontainers import SortedList


class RealTimeKalmanTracker(SimulatorObject):
    def __init__(self,id="Tr1",dt=0.1, std_acc=2.0, broker=None, color="b"):
        super().__init__(id,dt)
        self.broker=broker
        self.color=color
        self.tracker = TorchMultiObsKalmanFilter(dt,std_acc)

        self.log_publisher=Publisher("log",broker)
        self.track_publisher=Publisher("tracks",broker)
        self.detection_subscriber = Subscriber("detections",broker)
        self.control_subscriber=Subscriber("control",broker)

        #self.observation_history = SortedList(key=lambda x:x["t"])
        #self.kf_history = SortedList(key=lambda x:x["t"])

        self.kf_history_max_size=100
        self.observation_history_max_size=100

        self.last_update_time = None


    def step(self):
        
        #Collect new observations and insert in time order
        #Keep track of earliest observation using min_t
        z=[]; z_std=[]; t=time.time()
        min_t = self.last_update_time
        count=0
        while(self.detection_subscriber.have_msg()):
            msg  = self.detection_subscriber.pop()
            if(min_t is None or msg["t"] < min_t):
                min_t = np.floor(msg["t"]*10)/10
            self.observation_history.add(msg)

            if(len(self.observation_history)>self.observation_history_max_size):
                self.observation_history.pop(0)

            count=count+1

        #self.log_publisher.publish({"id":self.id, "msg":"Got %d new observations"%count})

        #If we have some observations, update the track
        if(count==0):
            self.tracker.predict()
        else:

            #Rewind the tracker to the last update before min_t
            if(self.last_update_time is not None):
                i = self.kf_history.bisect_left({"t":min_t})-1
                if(i>=0):
                    x = self.kf_history[i]["x"]
                    P = self.kf_history[i]["P"]
                    #Clear the history after this point
                    for j in np.arange(i+1,len(self.kf_history)):
                        self.kf_history.pop[-1]
                else:
                    x=np.matrix([[0], [0], [0], [0]])
                    P=np.eye(4)
                self.tracker.x=x
                self.tracker.P=P

            #Get observations since minimum time value of observations in this batch
            max_t      = self.observation_history[-1]["t"]
            num_chunks = int(np.ceil((max_t-min_t)/self.dt))

            #self.log_publisher.publish({"id":self.id, "msg":"Processing %d chunks"%num_chunks})

            #Re-process all observations since min_t and save into history
            for c in range(num_chunks):
                start_t = min_t + c*self.dt
                end_t   = min_t + (c+1)*self.dt - (1e-6)
                z=[]; z_cov=[]
                for obs in self.observation_history.irange({"t":start_t},{"t":end_t}):
                    pos  = obs["pos"]
                    cov  = obs["cov"]
                    t    = obs["t"]
                    z.append(pos)
                    z_cov.append(cov)   

                    #self.log_publisher.publish({"id":self.id, "msg":"Adding observation at t=%f to chunk %d from [%f to %f]"%(t,c,start_t,end_t) })


                self.tracker.predict()
                if(len(z)>0):
                    z = np.hstack(z)
                    #z_cov = np.hstack(z_cov)
                    self.tracker.update(z,z_cov,store=True)

                self.kf_history.add({"t":start_t,"x":self.tracker.x,"P":self.tracker.P})
                if(len(self.kf_history)>self.kf_history_max_size):
                    self.kf_history.pop(0)

        x = self.tracker.x.detach().numpy()
        P = self.tracker.P.detach().numpy()
        t = np.floor(time.time()*10)/10
        self.track_publisher.publish({"id":self.id, "pos":x, "cov":P, "col":self.color,"t":t})


class KalmanTracker(SimulatorObject):
    def __init__(self,id="Tr1",dt=0.1, std_acc=2, broker=None, color="b"):
        super().__init__(id,dt)
        self.broker=broker
        self.color=color
        self.tracker = MultiObsKalmanFilter(dt,std_acc)

        self.track_publisher=Publisher("tracks",broker)
        self.detection_subscriber = Subscriber("detections",broker)
        self.control_subscriber=Subscriber("control",broker)

    def step(self):

        #Run the Kalman filter forward
        self.tracker.predict()
        
        #Check to see if there are observations
        z=[]; z_std=[]; t=time.time()
        while(time.time()-t<self.dt/2):
            while(self.detection_subscriber.have_msg()):
                msg  = self.detection_subscriber.pop()
                pos  = msg["pos"]
                std = msg["std"]
                z.append(pos)
                z_std.append(std)
            time.sleep(self.dt/10)

        #If there are observations, update the track
        if(len(z)>0):
            z = np.hstack(z)
            z_std = np.hstack(z_std)
            self.tracker.update(z,z_std,store=True)

        x = np.array(self.tracker.x)[:2]
        self.track_publisher.publish({"id":self.id, "pos":x, "col":self.color})



class MultiObsKalmanFilter(object):
    def __init__(self, dt, std_acc):
        """
        :param dt: sampling time (time for 1 cycle)
        :param std_acc: process noise magnitude
        """

        # Define sampling time
        self.dt = dt

        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])

        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])


        # Define Measurement Mapping Matrix for one observation
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        #Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795

        # Update time state
        #x_k =Ax_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) 

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z, zcov,store=True):
        """
        :param z: shape (2,K) array of observations
        :param zcov: list of K 2x2 observation covariances
        """
        
        K = z.shape[1]
        
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        
        H = np.tile(self.H,[K,1])

        R = scipy.linalg.block_diag(*zcov)

        z = z.T.reshape([2*K,1])
        
        S = np.dot(H, np.dot(self.P, H.T)) + R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))  #Eq.(11)

        x = self.x + np.dot(K, (z - np.dot(H, self.x)))   #Eq.(12)

        I = np.eye(H.shape[1])

        #Update error covariance matrix
        #Note: The elementwise products here look wrong.
        #Should probably be matrix multiplications instead.
        P = (I - (K * H)) * self.P  + 0.001*np.eye(4)  #Eq.(13)
        
        if(store):
            self.x=x
            self.P=P
            
        return x[0:2],P



class TorchMultiObsKalmanFilter(torch.nn.Module):
    def __init__(self, dt, std_acc):
        """
        :param dt: sampling time (time for 1 cycle)
        :param std_acc: process noise magnitude
        """
        super(TorchMultiObsKalmanFilter, self).__init__()

        # Define sampling time
        self.dt = dt

        # Initial State
        x = torch.FloatTensor([[0], [0], [0], [0]])
        self.register_buffer('x', x)


        # Define the State Transition Matrix A
        A = torch.FloatTensor([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.register_buffer('A', A)


        # Define Measurement Mapping Matrix for one observation
        H = torch.FloatTensor([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.register_buffer('H', H)

        #Initial Process Noise Covariance
        std_acc_prime_init = np.log(np.exp(std_acc)-1)
        self.std_acc_prime = torch.nn.Parameter(torch.FloatTensor([std_acc_prime_init]))

        Q_base = torch.FloatTensor([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]])
        self.register_buffer('Q_base', Q_base)

        #Initial Covariance Matrix
        P = torch.eye(self.A.shape[1])
        self.register_buffer('P', P)

    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795

        # Update time state
        #x_k =Ax_(k-1)     Eq.(9)
        self.x = self.A @ self.x

        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        std_acc = torch.nn.functional.softplus(self.std_acc_prime)
        Q = self.Q_base*(std_acc**2)
        self.P = self.A @ self.P @ self.A.T + Q
        return self.x[0:2]

    def update(self, z, zcov, store=True):
        """
        :param z: shape (2,K) tensor of observations
        :param zcov: list of K 2x2 observation covariances
        """
        
        L = z.shape[1]
        
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        
        H = torch.tile(self.H,[L,1])

        #R = torch.block_diag(*[torch.FloatTensor(zc) for zc in zcov])
        R = torch.block_diag(*zcov)

        #z = torch.FloatTensor(z).T.reshape([2*L,1])
        z = z.t().reshape([2*L,1])

        S = H @ self.P @ H.T + R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = self.P @ H.T @ torch.inverse(S)  #Eq.(11)

        x = self.x + K @ (z - H@self.x)   #Eq.(12)

        I = torch.eye(H.shape[1]).to(H.device)
        
        # Update error covariance matrix
        P = (I - (K @ H)) @ self.P  + 0.00001*torch.eye(4).to(H.device)  #Eq.(13)
        
        if(store):
            self.x=x
            self.P=P
            
        return x[0:2],P

    def forward(self,z_pos, z_cov):
        '''
        Inputs:        
        * z_pos: a list of observation tensors of shape (2, D_t). The length of the list L is the number of 
        time points t in the tracking sequence. 2 is the dimensionality of the position vectors, 
        and D_t is the number of observations per time point t. The value of D_t can be different for different time
        points.
        * z_cov: a list of list of observation standard deviation matrices. The length of the list L
        is the length of the sequence. Each nested list has length D_t. Element j in each
        nested list is the 2x2 covariance tensor for the jth observation at time t.

        Outputs:
        * x_pos_hat: Tensor of shape (2,L) giving the KF estimated mean location of the tracked object at each time point L
        * x_cov_hat: Tensor of shape (2,2,L) giving the KF estimated covariance of the location of the tracked object at each time point
        '''

        #Start at mean of observations
        device = z_pos[0].device
        self.x = torch.FloatTensor([[0], [0], [0], [0]]).to(device)
        self.P = 100*torch.eye(self.A.shape[1]).to(device)

        L = len(z_pos)
        x_pos_hat = torch.zeros(2,L).to(device)
        x_cov_hat = torch.zeros(2,2,L).to(device)
        for i in range(L):
            self.update(z_pos[i],z_cov[i])
            x_pos_hat[:,i] = self.x[:2].flatten()
            x_cov_hat[:,:,i] = self.P[:2,:2]
            self.predict()

        return(x_pos_hat, x_cov_hat)

    def loss(self, x_pos, x_pos_hat, x_pos_cov):
        #Compute the average Gaussian NLL
        T=x_pos.shape[1]
        #nll =torch.zeros(1)
        nll = 0 
        for t in range(T):
            nll = nll + 0.5*torch.log(torch.det(2.0*np.pi*x_pos_cov[:,:,t])) + 0.5*(x_pos[:,[t]]-x_pos_hat[:,[t]]).T @ torch.inverse(x_pos_cov[:,:,t]) @ (x_pos[:,[t]]-x_pos_hat[:,[t]])
            #m = torch.distributions.multivariate_normal.MultivariateNormal(x_pos_hat[:,t], x_pos_cov[:,:,t])
            #nll = nll - m.log_prob(x_pos[:,t])
        return(nll/T)

    def fit(self,x_pos, z_pos, z_cov, max_iter=1000,lr=0.1):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for i in range(max_iter):
            
            x_pos_hat, x_pos_cov = self.forward(z_pos, z_cov)
            l = self.loss( x_pos, x_pos_hat, x_pos_cov)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


            with torch.no_grad():
                if(i%10==0):
                    std_acc = torch.nn.functional.softplus(self.std_acc_prime)
                    print("Iteration: %d: Loss: %.4f  std_acc: %.4f"%(i,l,std_acc))
