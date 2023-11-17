import time
import threading

class Broker:
    def __init__(self):
        self.topics=[]
        self.subscriptions={}
        self.queues={}
        self.read_heads={}

    def subscribe(self, topic, sub):
        if topic not in self.topics:
            self.topics.append(topic)
            self.queues[topic]=[]
            self.subscriptions[topic]=[sub]
        else:
            if(sub not in self.subscriptions[topic]):
                self.subscriptions[topic].append(sub)

    def lagged_publish(self,topic,msg,lag):
        time.sleep(lag)
        for sub in self.subscriptions[topic]:
            sub.push(msg)

    def publish(self, topic, msg,lag=0):
        if topic not in self.topics:
            self.topics.append(topic)        

        if(topic in self.subscriptions):
            if(lag==0):
                self.lagged_publish(topic,msg,0)
            else:
                thread = threading.Thread(target=self.lagged_publish,args=[topic,msg,lag])
                thread.start()

class Publisher:
    def __init__(self,topic,broker):
        self.broker=broker
        self.topic=topic

    def publish(self, msg, lag=0):
        self.broker.publish(self.topic,msg,lag)
        time.sleep(0.001) #Sleep to allow other events to run   

class Subscriber:
    def __init__(self,topic,broker):
        self.queue=[]
        broker.subscribe(topic,self)
    
    def push(self,msg):
        self.queue.append(msg)
        time.sleep(0.001) #Sleep to allow other events to run 

    def have_msg(self):
        time.sleep(0.001) #Sleep to allow other events to run 
        return len(self.queue)>0

    def pop(self,i=0):
        time.sleep(0.001) #Sleep to allow other events to run 
        if(len(self.queue)>0):
            return self.queue.pop(i)
        else:
            return None

    def clear(self):
        time.sleep(0.001) #Sleep to allow other events to run 
        self.queue=[]
