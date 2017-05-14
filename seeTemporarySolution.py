import numpy as np
import matplotlib.pyplot as plt


a = np.load("finalSolution.npy")
#dx = np.load("temporaryDX.npy")
#dy = np.load("temporaryDY.npy")

def plot(x,title):
    
    fig, axarr = plt.subplots(2, 2)
    img1 = axarr[0, 0].imshow(x[:,:,0].T, interpolation='nearest')
    axarr[0, 0].set_title(title[0])
    plt.axes(axarr[0,0])
    fig.colorbar(img1)
    
    img2 = axarr[0, 1].imshow(x[:,:,1].T, interpolation='nearest')
    axarr[0, 1].set_title(title[1])
    plt.axes(axarr[0,1])
    fig.colorbar(img2)
    
    img3 = axarr[1, 0].imshow(x[:,:,2].T, interpolation='nearest')
    axarr[1, 0].set_title(title[2])
    plt.axes(axarr[1,0])
    fig.colorbar(img3)
    
    
    img4 = axarr[1, 1].imshow(x[:,:,3].T, interpolation='nearest')
    axarr[1, 1].set_title(title[3])
    plt.axes(axarr[1,1])
    fig.colorbar(img4)
    # adjust spacing between subplots so that things don't overlap
    fig.tight_layout()

plot(a,["Re(Deltax)","Im(Deltax)","Re(Deltay)","Im(Deltay)"] )
#plot(dx,["dx Re(Deltax)","dx Im(Deltax)","dx Re(Deltay)","dx Im(Deltay)"] )
#plot(dy,["dy Re(Deltax)","dy Im(Deltax)","dy Re(Deltay)","dy Im(Deltay)"] )
plt.show()
