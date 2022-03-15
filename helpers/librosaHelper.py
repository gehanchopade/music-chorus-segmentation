# import librosa
# import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    import librosa
    import librosa.display
except:
    print("Run \"python3 install_requirements.py\" to install dependencies or follow README for setup instructions")
    sys.exit(42)

import warnings
warnings.filterwarnings("ignore")
class LibrosaHelpers:
    def __init__(self,path,sampling_rate=44100,show_shapes=False):
        self.path=path
        self.sampling_rate=sampling_rate
        self.show_shapes=show_shapes
        self.audio_arr=self.get_audio()
        self.mfcc=self.get_MFCC()
        self.tempo,self.beats=self.get_beats_tempo()
    def get_audio(self):
        """
        returns audio in the form of a numpy array using librosa.load
        """
        audio_arr,_=librosa.load(path=self.path,sr=self.sampling_rate)
        if(self.show_shapes):
            print(audio_arr.shape)
        return audio_arr
    def get_MFCC(self,n_mfcc=128,dct_type=2,hop_length=512,norm=None):
        """
        n_mfcc : int > 0 [scalar]
            number of MFCCs to return

        dct_type : {1, 2, 3}
            Discrete cosine transform (DCT) type.
            By default, DCT type-2 is used.

        norm : None or 'ortho'
            If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an ortho-normal
            DCT basis.
        hop_length : int > 0 [scalar]
            number of samples between successive frames.
            See `librosa.core.stft`

        """
        mfcc=librosa.feature.mfcc(y=self.audio_arr,sr=self.sampling_rate,n_mfcc=n_mfcc,dct_type=dct_type,hop_length=hop_length)
        if(self.show_shapes):
            print(mfcc.shape)
        return mfcc
    def get_beats_tempo(self):
        tempo, beats = librosa.beat.beat_track(y=self.audio_arr, sr=self.sampling_rate, trim=False)
        return tempo,beats
    def plot_waveform(self,audio,figsize=(20,8)):
        '''
        audio : numpy.array
            audio numpy array
        figsize : (int,int)
            figure size
        '''
        audio_arr = audio / np.max(np.abs(audio))
        plt.figure(figsize=figsize)
        plt.plot(np.linspace(0, len(audio_arr) / self.sampling_rate, num=len(audio_arr)), audio_arr)
        plt.show()
    def plot_spec(self,data,x_axis='time',y_axis='linear',figsize=(20,8)):
        '''
        data : numpy.array
            data numpy array
        x_axis : x_axis for spectrogram
        y_axis : y_axis for spectrogram
        figsize : (int,int)
            figure size
        '''
        plt.figure(figsize=figsize)
        librosa.display.specshow(data, sr=self.sampling_rate, x_axis=x_axis,y_axis=y_axis)
        plt.show()
    

