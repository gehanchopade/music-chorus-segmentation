from helpers.librosaHelper import LibrosaHelpers
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
class MusicCluster:
    def __init__(self,path):
        """
        path : Directory/File path
            Directory path should end in '/' ex: path='data/'
        isDir : True - path is a directory path
        
        """
        self.path=path
        self.librosa_helper=None
        self.spacer= lambda x=100: print("-"*x)
        self.preds=None
        self.mfcc=None
    def results(self,k,k_auto=False,preprocess=True,kplots=False):
        print("Clustering results for: "+self.path)
        self.librosa_helper=LibrosaHelpers(self.path)
        if(k_auto):
            k=self.getK(self.librosa_helper)
        self.mfcc=self.librosa_helper.mfcc
        if(preprocess):
            self.preprocess_data(self.librosa_helper)
        _=self.cluster(self.librosa_helper,k)
        self.display_plots(self.librosa_helper,kplots)
    
    def display_plots(self, librosa_helper,kplots):
        
        librosa_helper.plot_waveform(librosa_helper.audio_arr)
        self.spacer()
        librosa_helper.plot_spec(librosa_helper.mfcc)
        self.spacer()
        _=self.silhouette_score(10,librosa_helper,kplots)
        
        _=self.elbow_scores(10,librosa_helper,kplots)
        self.spacer
        self.plot_preds(librosa_helper)
    def elbow_scores(self,kmax,librosa_helper,plot=False):
        scores=[]
        for k in range(2,kmax+1):
            model=KMeans(n_clusters=k).fit(librosa_helper.mfcc)
            scores.append(model.inertia_)
        if(plot):
            plt.plot(range(2,kmax+1),scores)
            plt.show()
            self.spacer()
        return scores
    def silhouette_score(self,kmax,librosa_helper,plot=False):
        scores=[]
        for k in range(2,kmax+1):
            model=KMeans(n_clusters=k).fit(librosa_helper.mfcc)
            preds=model.labels_
            scores.append(silhouette_score(librosa_helper.mfcc,labels=preds,metric='euclidean'))
        if(plot):
            plt.plot(range(2,kmax+1),scores)
            plt.show()
            self.spacer()
        return scores

    def preprocess_data(self,librosa_helper):
        scale=preprocessing.StandardScaler()
        self.mfcc=scale.fit_transform(librosa_helper.mfcc)
    def plot_preds(self,librosa_helper):
        
        plt.figure(figsize=(20,8))
        sec_preds=[]
#         print(self.preds.shape,librosa_helper.audio_arr.shape,librosa_helper.sampling_rate,librosa_helper.mfcc.shape)
        secs=int(librosa_helper.audio_arr.shape[0]/librosa_helper.sampling_rate)
        for i in range(0,len(self.preds),int(self.preds.shape[0]/secs)):
            arr=self.preds[i:i+int(self.preds.shape[0]/(librosa_helper.audio_arr.shape[0]/librosa_helper.sampling_rate))]
            sec_preds.append(round(arr.mean()))
        
        sec_preds=np.array(sec_preds)
#         sec_preds[np.where(sec_preds==0)]=-1
        plt.scatter(np.linspace(0, len(sec_preds), num=len(sec_preds)),sec_preds,s=10,c='red',zorder=2)
        plt.plot(np.linspace(0, len(librosa_helper.audio_arr) / librosa_helper.sampling_rate, num=len(librosa_helper.audio_arr)), librosa_helper.audio_arr,zorder=1)
        plt.grid(True)
        plt.show()
    def cluster(self,librosa_helper,k):
        self.kmeans=KMeans(n_clusters=k)
        self.preds=self.kmeans.fit_predict(librosa_helper.mfcc.T)
        print("Clustering complete!!")
        return self.preds
    def getK(self,librosa_helper):
        k=self.silhouette_score(10,librosa_helper).index(max(self.silhouette_score(10,librosa_helper)))+2
        print("Best k: "+str(k))
        return k


# if __name__=="__main__":
#     musicCluster=MusicCluster()