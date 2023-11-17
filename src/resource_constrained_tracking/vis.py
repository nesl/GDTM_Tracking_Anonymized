import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, transforms
from pubsub import Subscriber
import numpy as np
import time

def confidence_ellipse(pos,cov,ax,color="b"):

    n_std=2

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=color,alpha=0.25)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = pos[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = pos[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf+ ax.transData)

    return(ellipse)

class TrackingVis():
    def __init__(self,figsize,env,broker):
        self.colors={}
        self.last_pos={}
        self.sensors={}
        self.handles={}
        self.types={}
        self.max_age={"detection":0.5, "track":1}
        self.last_perf=None
        self.first_time=None

        #self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)

        self.fig = plt.figure(constrained_layout=True,figsize=figsize)
        gs = plt.GridSpec(2, 2, figure=self.fig,height_ratios=[3,1])
        
        # create sub plots as grid
        ax0 = self.fig.add_subplot(gs[0, 0])
        ax1 = self.fig.add_subplot(gs[0, 1])
        ax2 = self.fig.add_subplot(gs[1, :])

        #self.fig.tight_layout()

        self.axes=[ax0,ax1,ax2]

        self.log_subscriber = Subscriber("log",broker)
        self.target_subscriber = Subscriber("targets",broker)
        self.sensor_subscriber = Subscriber("sensors",broker)
        self.detection_subscriber = Subscriber("detections",broker)
        self.track_subscriber = Subscriber("tracks",broker)
        self.performance_subscriber = Subscriber("performance",broker)

        for ax in self.axes[:2]:
            ax.set_xlim([env.minx-2,env.maxx+2])
            ax.set_ylim([env.miny-2,env.maxy+2])
            ax.grid(True)

        ax2.grid(True)
        ax2.set_ylim([0,10])
        ax2.set_xlim([0,2])

    def process_targets(self):
        t=time.time()
        while(self.target_subscriber.have_msg() and time.time()-t<0.1):
            msg  = self.target_subscriber.pop()
            id   = msg["id"]
            pos  = msg["pos"]
            color = msg["col"]

            if not id in self.handles:
                self.handles[id]=[]

            t=time.time()
            ages = [2,2]
            for i,ax in enumerate(self.axes[:2]):
                h=ax.plot(pos[0], pos[1],".",color=color)
                self.handles[id].append([t,ages[i],h])
                if(id in self.last_pos):
                    h=ax.plot([pos[0],self.last_pos[id][0]], [pos[1],self.last_pos[id][1]],"-",color=color,alpha=0.5)
                    self.handles[id].append([t,ages[i],h])

            self.last_pos[id]=pos

    def process_tracks(self):
        t=time.time()
        while(self.track_subscriber.have_msg() and time.time()-t<0.1):
            msg  = self.track_subscriber.pop()
            id   = msg["id"]
            pos  = np.array(msg["pos"][:2,0]).flatten()
            color = msg["col"]
            cov= msg["cov"][:2,:2]

            timeout=2
            if id not in self.handles:
                self.handles[id]=[]

            t=time.time()
            h=self.axes[1].plot(pos[0], pos[1],".",color=color)
            self.handles[id].append([t,timeout,h])
            if(id in self.last_pos):
                h=self.axes[1].plot([pos[0],self.last_pos[id][0]], [pos[1],self.last_pos[id][1]],"-",color=color,alpha=0.5)
                self.handles[id].append([t,timeout,h])
            self.last_pos[id]=pos

            ell = confidence_ellipse(pos,cov,self.axes[1],color=color)
            h=self.axes[1].add_artist(ell)
            self.handles[id].append([t,timeout,h])

    def process_sensors(self):
        t=time.time()
        while(self.sensor_subscriber.have_msg() and time.time()-t<0.1):
            msg  = self.sensor_subscriber.pop()
            id   = msg["id"]
            typ  = msg["type"]
            pos  = msg["pos"]
            color = msg["col"]
            self.sensors[id]=pos

            for ax in self.axes[:2]:
                plt.sca(ax)
                plt.plot(pos[0], pos[1],"o",color=color)
                plt.text(pos[0], pos[1]+1,id,horizontalalignment='center', verticalalignment='center')

    def process_detections(self):
        t=time.time()
        while(self.detection_subscriber.have_msg() and time.time()-t<0.1):
            msg  = self.detection_subscriber.pop()
            id   = msg["id"]
            pos  = np.array(msg["pos"]).flatten()
            true_pos  = np.array(msg["true_pos"]).flatten()
            color = msg["col"]
            cov = msg["cov"]

            self.remove(id,remove_all=True)
            h=self.axes[0].plot([self.sensors[id][0],pos[0]], [self.sensors[id][1],pos[1]],".:",color=color,alpha=0.5)
            self.handles[id].append([time.time(),0.5,h])
            
            ell = confidence_ellipse(true_pos,cov,self.axes[0],color=color)
            h=self.axes[0].add_artist(ell)
            self.handles[id].append([time.time(),0.5,h])

    def remove(self,id,remove_all=False):
        new_list=[]
        if(id in self.handles):
            for i in range(len(self.handles[id])):
                t,a,h = self.handles[id][i]

                if(not remove_all and time.time()-t<a):
                    new_list.append([t,a,h])
                else:
                    if(type(h)==list):
                        for hh in h:
                            hh.remove()
                    else:
                        h.remove()
        self.handles[id]=new_list         

    def process_stale(self):
        for id in self.handles:
            self.remove(id)

    def process_log(self):
        t=time.time()
        while(self.log_subscriber.have_msg() and time.time()-t<0.1):
            msg  = self.log_subscriber.pop()
            id   = msg["id"]
            txt  = msg["msg"] 
            print("[%s]: %s"%(str(id),txt))

    def process_performance(self):
        while(self.performance_subscriber.have_msg()):
            msg  = self.performance_subscriber.pop()
            id=msg["id"]
            if(self.first_time is None):
                self.first_time=np.floor(10*(msg["t"]))/10
                t=0
            else:
                t=np.floor(10*(msg["t"]-self.first_time))/10
            msg["t"]=t

            metrics = ["imae","cmae"]
            colors  = ["b","r"] 



            custom_lines =[]
            for i,m in enumerate(metrics):
                custom_lines.append(plt.Line2D([0],[0],color=colors[i],lw=1))

            if(id not in self.handles): self.handles[id]=[]
            for i,m in enumerate(metrics):
                h=self.axes[2].plot(msg["t"],msg[m],'.',color=colors[i])
                self.handles[id].append([time.time(),2,h])
                if(self.last_perf is not None):
                    h=self.axes[2].plot([self.last_perf["t"],msg["t"]],[self.last_perf[m],msg[m]],'-',color=colors[i])
                    self.handles[id].append([time.time(),2,h])

            self.last_perf=msg
            self.axes[2].set_xlim([max(0,msg["t"]-2),max(2,msg["t"])])
            self.axes[2].legend(custom_lines,["Instantaneous MAE","Time Average MAE"],loc='upper left')
            self.axes[2].set_xticks(np.linspace(max(0,msg["t"]-2), max(2,msg["t"]),21 ))

    def update(self):

        self.process_log()
        self.process_stale()
        self.process_sensors()
        self.process_targets()
        self.process_detections()
        self.process_tracks()
        self.process_performance()
        self.fig.canvas.draw()

