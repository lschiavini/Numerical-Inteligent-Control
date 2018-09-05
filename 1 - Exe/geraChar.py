#% GERACHAR - ANN OCR 
#% 
#% ICIN/UnB Ago/2018 - Adolfo Bauchspiess
#% Generate Training Patterns 9x7


import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.lines as lines


def geraChar():
    P=np.zeros((63,16),dtype=int);
    T=np.eye(16,dtype=int);
    P[:,0]=[			#L
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1
    ]
    P[:,1]=[			#M
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,0,0,0,1,1,
    1,0,1,0,1,0,1,
    1,0,0,1,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1
    ]
    P[:,2]=[			#N
    1,0,0,0,0,0,1,
    1,1,0,0,0,0,1,
    1,0,1,0,0,0,1,
    1,0,0,1,0,0,1,
    1,0,0,0,1,0,1,
    1,0,0,0,0,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1
    ]
    P[:,3]=[			#O
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0
    ]
    P[:,4]=[			#P
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0,
    1,0,0,0,0,0,0
    ]
    P[:,5]=[			#Q
    1,1,1,1,1,1,1,
    1,1,0,0,0,1,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,0,0,0,0,0,1,
    1,1,0,0,0,1,1,
    1,1,1,1,1,1,1,
    0,0,0,1,1,0,0
    ]
    P[:,6]=[		#R
    1,1,1,1,1,1,1,
    1,1,0,0,0,0,1,
    1,1,0,0,0,0,1,
    1,1,0,0,0,0,1,
    1,1,1,1,1,1,1,
    1,1,0,1,0,0,0,
    1,1,0,0,1,0,0,
    1,1,0,0,0,1,0,
    1,1,0,0,0,0,1
    ]
    P[:,7]=[		#S
    0,0,1,1,1,1,0,
    0,1,0,0,0,0,1,
    0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,1,0,0,0,
    1,0,0,0,1,0,0,
    1,0,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0
    ]
    P[:,8]=[			#T
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0,
    0,0,1,1,1,0,0
    ]
    P[:,9]=[			#U
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,1,1,1,1,1,
    1,1,1,1,1,1,1
    ]
    P[:,10]=[			#V
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    0,1,0,0,0,1,0,
    0,1,1,0,1,1,0,
    0,1,0,1,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,1,0,0,0
    ]
    P[:,11]=[			#Y
    1,0,0,0,0,0,1,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0
    ]
    P[:,12]=[			#X
    1,0,0,0,0,0,1,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,1,0,0,
    0,1,0,0,0,1,0,
    1,0,0,0,0,0,1
    ]
    P[:,13]=[			#Z
    1,1,1,1,1,1,1,
    0,0,0,0,0,1,1,
    0,0,0,0,0,1,1,
    0,0,0,0,1,1,0,
    0,0,0,1,1,0,0,
    0,0,1,1,0,0,0,
    1,1,1,0,0,0,0,
    1,1,0,0,0,0,0,
    1,1,1,1,1,1,1
    ]
    P[:,14]=[			#0
    0,0,1,1,1,0,0,
    0,1,1,1,1,1,0,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,1,0,1,1,
    1,1,0,0,0,1,1,
    1,1,0,0,0,1,1,
    0,1,1,1,1,1,0,
    0,0,1,1,1,0,0
    ]
    P[:,15]=[			#1
    0,0,0,1,0,0,0,
    0,0,1,1,0,0,0,
    0,1,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0
    ]

    P=P.transpose()
    
    return (P,T)


def gchar_ruido(P, Ruido):
# Adds noide to the pattern P
# Returns the noise pattern Pn

    np.random.seed(int(time.clock()))

    Pn=np.zeros((63,16),dtype=int)
    
    # Copy function! P is only a pointer!!
    Pn=copiaV(P)
    
    BitsRuido=int(np.ceil(63*Ruido/100))
    noise_ind=np.zeros((BitsRuido,1),dtype=int)

    if BitsRuido == 0:
        return P
    
    for l in range(16):
        vals = np.random.rand(BitsRuido,1)*63
        for k in range(BitsRuido):
            noise_ind[k] = int(np.floor(vals[k]))
           
        for i in range(BitsRuido): 
            if Pn[l,noise_ind[i]] == 0: Pn[l,noise_ind[i]] = 1
            else: Pn[l,noise_ind[i]] = 0
        
    return Pn

def copia(x):
    y=x
    return y

def copiaV(P):
    f=np.vectorize(copia)
    return f(P)

def complement(x):
    return -x+1

def CompP(P):
    f=np.vectorize(complement)
    return f(P)

def showChar(P):
# Displays the 16 9x7 Characters

    plt.figure(figsize=(18,14))
    for i in range(16):
        plt.subplot(1,16,i+1)        
        Pc=CompP(P)
        plt.imshow(Pc[i,:].reshape(9,7),cmap=cm.Greys_r)
        plt.axis('off')    
    plt.show()
    return

def validacao(Pn,net):
# calculate the output of the net for a given noise Patterns Pn
# bincon - incorrect Chars; idx - the recognized pattern (should be range(16))

    m=[]
    idx=[]

    # calculate predictions
    predictions = net.predict(Pn)
    for k in range(16):
       p = predictions[k,:]
       m.append(np.max(p))
       idx.append(np.argmax(p))
    
    # calculate number or incorrect characters
    bincor = 0; 
    for i in range(16): 
        if idx[i] <> i: bincor +=1;
            
    return (bincor,idx)   
