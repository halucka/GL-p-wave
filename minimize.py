import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import time
import random

start_time = time.time()

# lattice parameters

nx = 31    # will be used globally
ny = 31     # must be both odd for TiltedEdge, nx>~ny
hx = 0.1
hy = 0.1

# parallel hole is a rectangle centered on [(nx-1)/2, (ny-1)/2]
hasParallelHole = False
hasTiltedHole = False
hasTiltedEdge = True


if(hasParallelHole):
    sx = 5 # hole size
    sy = 5
    
    lowX = (nx-sx)//2 ## watch out: integer division (make sure nx ~ sx (mod 2) & ny ~ sy (mod 2) to avoid unexpected behavior)
    highX = (nx+sx)//2

    lowY = (ny-sy)//2
    highY = (ny+sy)//2

    rangeX = slice(lowX,highX) # defined so that it has sx points included
    rangeY = slice(lowY,highY)


if(hasTiltedHole):
    tx = 3
    ty = 5
    lx = 2*tx + 1
    ly = 2*ty + 1


if(hasTiltedEdge):
    lowY = np.empty(nx)
    highY = np.empty(nx)
    lowX = np.empty(ny)
    highX = np.empty(ny)
    
    rangeY = []
    emptyRangeY = []
    
    rangeX = []
    emptyRangeX = []
    

    for x in range(0, nx):
        lowY[x] = 0
        highY[x] = np.minimum(ny,2*x+1)
    lowY = lowY.astype(np.int64)
    highY = highY.astype(np.int64)  # need to explicitly make it integer

    for x in range(0, nx):
        rangeY.append(slice(lowY[x],highY[x]))
        emptyRangeY.append(slice(highY[x],ny))


    for y in range(0,ny):
        lowX[y] = np.minimum((y+1)//2,nx) ### floor division
        highX[y] = nx
    lowX = lowX.astype(np.int64) # need to explicitly make it integer
    highX = highX.astype(np.int64)

    for y in range(0,ny):
        rangeX.append(slice(lowX[y],highX[y])) ## all points that belong to the mesh at that row
        emptyRangeX.append(slice(0,lowX[y]))

# p-wave GL free energy parameters
# at the moment chosen to give bulk norm 1 for both Delta_x and Delta_y

alpha = -4
Beta1 = 1
Beta2 = 1
K1 = 1
K2 = 1
K3 = 1

# starting point of minimization
a = np.random.random((nx,ny,4))

#a = np.load("temporarySolution.npy")

##initialize to homogeneous px + i py
#a = np.zeros((nx,ny,4))
#a[:,:,0] = 1
#a[:,:,3] = 1

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



def DX(x):
    #dx = np.gradient(x[:,:,:], axis=0)
    
    dx = np.zeros((nx,ny,4))
    
    dx[0,:,:] = (-25*x[0,:,:]+48*x[1,:,:]-36*x[2,:,:]+16*x[3,:,:]-3*x[4,:,:])/12.0
    #x[1,:,:]-x[0,:,:]
    dx[1,:,:] = (-3*x[0,:,:]-10*x[1,:,:]+18*x[2,:,:]-6*x[3,:,:]+x[4,:,:])/12.0
    # x[2,:,:]-x[1,:,:]

    dx[2:-2,:,:] = (x[0:-4,:,:]-8*x[1:-3,:,:]+8*x[3:-1,:,:]-x[4:,:,:])/12.0
    

    dx[-2,:,:] = (-x[-5,:,:]+6*x[-4,:,:]-18*x[-3,:,:]+10*x[-2,:,:]+3*x[-1,:,:])/12.0
    #x[-2,:,:]-x[-3,:,:]
    dx[-1,:,:] = (3*x[-5,:,:]-16*x[-4,:,:]+36*x[-3,:,:]-48*x[-2,:,:]+25*x[-1,:,:])/12.0
    #x[-1,:,:]-x[-2,:,:]


    dx[0,:,:] = x[1,:,:]-x[0,:,:]
    dx[1:-1,:,:] = 0.5*(x[2:,:,:]-x[0:-2,:,:])
    dx[-1,:,:] = x[-2,:,:]-x[-1,:,:]

    if(hasParallelHole):
        dx[rangeX,rangeY,:] = 0
        
        # fix dx on AC and BD edges
        
        # AC edge
        dx[lowX -1,rangeY,:] = (3*x[lowX-5,rangeY,:]-16*x[lowX-4,rangeY,:]+36*x[lowX-3,rangeY,:]-48*x[lowX-2,rangeY,:]+25*x[lowX-1,rangeY,:])/12.0
        #x[lowX -1,rangeY,:] - x[lowX -2, rangeY,:]
        dx[lowX -2,rangeY,:] = (-x[lowX-5,rangeY,:]+6*x[lowX-4,rangeY,:]-18*x[lowX-3,rangeY,:]+10*x[lowX-2,rangeY,:]+3*x[lowX-1,rangeY,:])/12.0

        # BD edge
        dx[highX,rangeY,:] = (-25*x[highX,rangeY,:]+48*x[highX+1,rangeY,:]-36*x[highX+2,rangeY,:]+16*x[highX+3,rangeY,:]-3*x[highX+4,rangeY,:])/12.0
        #x[highX +1,rangeY,:] - x[highX,rangeY,:]
        dx[highX+1,rangeY,:] = (-3*x[highX,rangeY,:]-10*x[highX+1,rangeY,:]+18*x[highX+2,rangeY,:]-6*x[highX+3,rangeY,:]+x[highX+4,rangeY,:])/12.0
    
    if(hasTiltedEdge):
        # make a hole
        for ypos in range(0,ny):
            dx[emptyRangeX[ypos],ypos,:] = 0
                    # lowX[ypos] is the east-most non-zero element
            dx[lowX[ypos],ypos,:] = (-25*x[lowX[ypos],ypos,:]+48*x[lowX[ypos]+1,ypos,:]-36*x[lowX[ypos]+2,ypos,:]+16*x[lowX[ypos]+3,ypos,:]-3*x[lowX[ypos]+4,ypos,:])/12.0
            dx[lowX[ypos]+1,ypos,:] = (-3*x[lowX[ypos],ypos,:]-10*x[lowX[ypos]+1,ypos,:]+18*x[lowX[ypos]+2,ypos,:]-6*x[lowX[ypos]+3,ypos,:]+x[lowX[ypos]+4,ypos,:])/12.0


    return dx/hx

def DY(x):
    #dy = np.gradient(x[:,:,:], axis=1)/hy

    dy = np.zeros((nx,ny,4))

    dy[:,0,:] = (-25*x[:,0,:]+48*x[:,1,:]-36*x[:,2,:]+16*x[:,3,:]-3*x[:,4,:])/12.0
    #x[:,1,:]-x[:,0,:]
    dy[:,1,:] = (-3*x[:,0,:]-10*x[:,1,:]+18*x[:,2,:]-6*x[:,3,:]+x[:,4,:])/12.0
    #x[:,2,:]-x[:,1,:]

    dy[:,2:-2,:] = (x[:,0:-4,:]-8*x[:,1:-3,:]+8*x[:,3:-1,:]-x[:,4:,:])/12.0
    
    dy[:,-2,:] = (-x[:,-5,:]+6*x[:,-4,:]-18*x[:,-3,:]+10*x[:,-2,:]+3*x[:,-1,:])/12.0
    #x[:,-3,:]-x[:,-2,:]
    dy[:,-1,:] = (3*x[:,-5,:]-16*x[:,-4,:]+36*x[:,-3,:]-48*x[:,-2,:]+25*x[:,-1,:])/12.0
    #x[:,-2,:]-x[:,-1,:]

    
    
#    dy[:,0,:] = x[:,1,:]-x[:,0,:]
#    dy[:,1:-1,:] = 0.5*(x[:,2:,:]-x[:,0:-2,:])
#    dy[:,-1,:] = x[:,-2,:]-x[:,-1,:]

    if(hasParallelHole):
        dy[rangeX,rangeY,:] = 0
        
        # fix dy on AB and CD edges
        
        # AB edge
        dy[rangeX, highY,:] = (-25*x[rangeX,highY,:]+48*x[rangeX,highY+1,:]-36*x[rangeX,highY+2,:]+16*x[rangeX,highY+3,:]-3*x[rangeX,highY+4,:])/12.0
        dy[rangeX, highY+1,:] = (-3*x[rangeX,highY,:]-10*x[rangeX,highY+1,:]+18*x[rangeX,highY+2,:]-6*x[rangeX,highY+3,:]+x[rangeX,highY+4,:])/12.0
        #x[rangeX, highY +1,:] -x[rangeX, highY,:]
        
        
        # CD edge
        dy[rangeX, lowY-1,:] = (3*x[rangeX,lowY-5,:]-16*x[rangeX,lowY-4,:]+36*x[rangeX,lowY-3,:]-48*x[rangeX,lowY-2,:]+25*x[rangeX,lowY-1,:])/12.0
        dy[rangeX,lowY-2,:] = (-x[rangeX,lowY-5,:]+6*x[rangeX,lowY-4,:]-18*x[rangeX,lowY-3,:]+10*x[rangeX,lowY-2,:]+3*x[rangeX,lowY-1,:])/12.0
        #x[rangeX, lowY -1,:] -x[rangeX, lowY -2,:]
    
    if(hasTiltedEdge):
        for ypos in range (0,ny):
            dy[emptyRangeX[ypos],ypos,:] = 0

            if(ypos==0):
                dy[0,0,:] = 0       # actual sharp corner #################################
                dy[1,0,:] = (-3*x[1,0,:]+4*x[1,1,:]-x[1,2,:])/2.0           # point C
            if(ypos==1):
                dy[1,1,:] = (x[1,2,:]-x[1,0,:])/2.0                         # point A
            if(ypos==2):
                dy[1,2,:] = (x[1,0,:]-4*x[1,1,:]+3*x[1,2,:])/2.0            # point B
            if(ypos>2):
                if(ypos % 2 == 1): #odd
                    dy[lowX[ypos],ypos,:] = (-x[lowX[ypos],ypos-3,:]+6*x[lowX[ypos],ypos-2,:]-18*x[lowX[ypos],ypos-1,:]+10*x[lowX[ypos],ypos,:]+3*x[lowX[ypos],ypos+1,:])/12.0
                if(ypos % 2 == 0): #even
                    dy[lowX[ypos],ypos,:] = (3*x[lowX[ypos],ypos-4,:]-16*x[lowX[ypos],ypos-3,:]+36*x[lowX[ypos],ypos-2,:]-48*x[lowX[ypos],ypos-1,:]+25*x[lowX[ypos],ypos,:])/12.0


    return dy/hy


# define free energy function

def modFreeEnergy(x):
    
    x = x.reshape((nx,ny,4)) # because scipy.optimize.minimize casts x into (nx*ny*4,)  vector
    
    #apply boundary conditions first
    
    # south
    x[:,0,2] = 0
    x[:,0,3] = 0
    x[:,0,0] = (48*x[:,1,0]-36*x[:,2,0]+16*x[:,3,0]-3*x[:,4,0])/25.0
    x[:,0,1] = (48*x[:,1,1]-36*x[:,2,1]+16*x[:,3,1]-3*x[:,4,1])/25.0
    
    # north
    x[:,-1,2] = 0
    x[:,-1,3] = 0
    x[:,-1,0] = (-3*x[:,-5,0]+16*x[:,-4,0]-36*x[:,-3,0]+48*x[:,-2,0])/25.0
    x[:,-1,1] = (-3*x[:,-5,1]+16*x[:,-4,1]-36*x[:,-3,1]+48*x[:,-2,1])/25.0
    
    # west
    x[0,:,0] = 0
    x[0,:,1] = 0
    x[0,:,2] = (48*x[1,:,2]-36*x[2,:,2]+16*x[3,:,2]-3*x[4,:,2])/25.0        # to have d(Delta_y)/dx =0
    x[0,:,3] = (48*x[1,:,3]-36*x[2,:,3]+16*x[3,:,3]-3*x[4,:,3])/25.0
    
    # east
    x[-1,:,0] = 0
    x[-1,:,1] = 0
    x[-1,:,2] = (-3*x[-5,:,2]+16*x[-4,:,2]-36*x[-3,:,2]+48*x[-2,:,2])/25.0
    x[-1,:,3] = (-3*x[-5,:,3]+16*x[-4,:,3]-36*x[-3,:,3]+48*x[-2,:,3])/25.0
    
    if(hasParallelHole):
        # make a hole
        x[rangeX,rangeY,:] = 0

        # boundary conditions on the hole
        # AB edge
        x[rangeX,highY,2:4] = 0
        x[rangeX,highY,0:2] = (48*x[rangeX,highY+1,0:2]-36*x[rangeX,highY+2,0:2]+16*x[rangeX,highY+3,0:2]-3*x[rangeX,highY+4,0:2])/25.0
        #x[rangeX,highY +1,0:2]
        
        # CD edge
        x[rangeX,lowY-1,2:4] = 0
        x[rangeX,lowY-1,0:2] = (-3*x[rangeX,lowY-5,0:2]+16*x[rangeX,lowY-4,0:2]-36*x[rangeX,lowY-3,0:2]+48*x[rangeX,lowY-2,0:2])/25.0
        #x[rangeX,lowY-2,0:2]

        # AC edge
        x[lowX-1,rangeY,0:2] = 0
        x[lowX-1,rangeY,2:4] = (-3*x[lowX-5,rangeY,2:4]+16*x[lowX-4,rangeY,2:4]-36*x[lowX-3,rangeY,2:4]+48*x[lowX-2,rangeY,2:4])/25.0
            #x[lowX-2,rangeY,2:4]

        # BD edge
        x[highX,rangeY,0:2] = 0
        x[highX,rangeY,2:4] = (48*x[highX+1,rangeY,2:4]-36*x[highX+2,rangeY,2:4]+16*x[highX+3,rangeY,2:4]-3*x[highX+4,rangeY,2:4])/25.0
            #x[highX +1,rangeY,2:4]
    
    if(hasTiltedHole):
        # make a hole
        for xpos in range(-(ly+tx)/2+1 , (ly+tx)/2):
            x[xpos +(nx-1)/2 , np.maximum(-2*xpos - ly -tx +2,(xpos-(ly+tx)/2 +1)/2) + (ny-1)/2 : np.minimum((xpos + (ly+tx)/2 -1)/2, -2*xpos + ly + tx -2 ) +1  + (ny-1)/2,:] = 0

    if(hasTiltedEdge):
        # make a hole
        #for xpos in range(0, nx):
        #    x[xpos,emptyRangeY[xpos],:] = 0
        for ypos in range(0,ny):
            #x[emptyRangeX[ypos],ypos,:] = 0 ### THIS DOESN'T WORK AND I DON'T KNOW WHY
            x[0:lowX[ypos]-1,ypos,:] = 0   ### THIS WORKS BUT I DON'T KNOW WHY
        #x[0:lowX[ypos],ypos,:] = 0
        # boundary conditions
        for ypos in range(0,ny):
            if(ypos==0):
                x[0,0,:] = (48*x[1,0,:]-36*x[2,0,:]+16*x[3,0,:]-3*x[4,0,:])/25.0
                x[0,0,2:4] = (2*hx/hy)*x[0,0,0:2]
                #x[0,0,:] = 0 # would follow from 1st equation only, applied from 2 directions
                x[1,0,2:4] = 0
                x[1,0,0:2] = (4*x[1,1,0:2]-x[1,2,0:2])/3.0
            if(ypos>0):
                xpos = lowX[ypos]
                x[xpos,ypos,0:2]=(x[xpos+2,ypos-1,0:2]*hy*hy+2*hx*hy*x[xpos+2,ypos-1,2:4])/(4*hx*hx+hy*hy)
                x[xpos,ypos,2:4]=(2*hx/hy)*x[xpos,ypos,0:2]
            
#            if(ypos==1):
#                x[1,1,:] = ((48*x[2,1,:]-36*x[3,1,:]+16*x[4,1,:]-3*x[5,1,:])*hy-3*hx*(x[1,2,:]-x[1,0,:]))/(25*hy)
#                x[1,1,2:4] = 2*x[1,1,0:2]
#            if(ypos==2):
#                x[1,2,:] = ((48*x[2,2,:]-36*x[3,2,:]+16*x[4,2,:]-3*x[5,2,:])*hy-3*hx*(x[1,0,:]-4*x[1,1,:]))/(9*hx+25*hy)
#                x[1,2,2:4] = 2*x[1,2,0:2]
#            if(ypos>2):
#                xpos = lowX[ypos]
##                B = 48*x[xpos+1,ypos,:]-36*x[xpos+2,ypos,:]+16*x[xpos+3,ypos,:]-3*x[xpos+4,ypos,:]
##                if(ypos % 2 == 1): #odd
##                    A1 = -x[xpos,ypos-3,:] +6*x[xpos,ypos-2,:]-18*x[xpos,ypos-1,:]+3*x[xpos,ypos+1,:] ### problematic for nx neq ny, probably need to implement the last few ypos manually!!!!!
##                    x[xpos,ypos,:] = (2*hy*B-hx*A1)/(10*hx+50*hy)
#                if(ypos % 2 == 0): #even
##                    A2 = 3*x[xpos,ypos-4,:]-16*x[xpos,ypos-3,:]+36*x[xpos,ypos-2,:]-48*x[xpos,ypos-1,:]
##                    x[xpos,ypos,:] = (2*hy*B-hx*A2)/(25*hx+50*hy)
#                    x[xpos,ypos,0:2] = (4*hy*x[xpos+1,ypos,2:4]+2*hy*x[xpos+1,ypos,0:2]-2*hx*x[xpos,ypos-1,2:4]-hx*x[xpos,ypos-1,0:2])/(10.0*hy-5.0*hx)
#                x[xpos,ypos,2:4] = 2*x[xpos,ypos,0:2] # Delta_y = 2*Delta_x
#


    # non-gradient terms
    alphaTerm = np.sum(x[:,:,:]**2) #  Note: 0:4 means 0 to 3
    alphaTerm *= alpha
    Beta1Term = np.sum((x[:,:,0]**2-x[:,:,1]**2+x[:,:,2]**2-x[:,:,3]**2)**2+4*(x[:,:,0]*x[:,:,1]+x[:,:,2]*x[:,:,3])**2)
    Beta1Term *= Beta1
    Beta2Term = np.sum((x[:,:,0]**2 + x[:,:,1]**2 + x[:,:,2]**2 + x[:,:,3]**2)**2)
    Beta2Term *= Beta2
    
    # gradients
    dx = DX(x)
    dy = DY(x)
    
    
    

    #  gradient terms
    K1Term = np.sum(dx[:,:,:]**2) + np.sum(dy[:,:,:]**2)
    K1Term *= K1
    
    K2Term = np.sum((dx[:,:,0]+dy[:,:,2])**2 + (dx[:,:,1]+dy[:,:,3])**2)
    K2Term *= K2
    
    K3Term = np.sum(dx[:,:,0]**2 + dx[:,:,1]**2 + dy[:,:,2]**2 + dy[:,:,3]**2 + 2*dx[:,:,2]*dy[:,:,0] + 2*dx[:,:,3]*dy[:,:,1])
    K3Term *= K3
    freeEnergy = (alphaTerm + Beta1Term + Beta2Term + K1Term + K2Term + K3Term)*hx*hy
    
    
    return freeEnergy

def afterEachIteration(x):
    # saving solution in Python format, get back i.e. by a = np.load("finalSolution.npy")
    x = x.reshape((nx,ny,4))
    np.save('temporarySolution', x)
    np.save('temporaryDX',DX(x))
    np.save('temporaryDY',DY(x))

# for setting up bounds
# bnds = [(-1,0)] * nx*ny*4 #((0, None), (0, None), (0, None))
# ,bounds=bnds

res = scipy.optimize.minimize(fun=modFreeEnergy, x0=a, method="L-BFGS-B",callback=afterEachIteration,options={'disp': True,'maxfun': 150000000}) #method="L-BFGS-B"
#res = scipy.optimize.minimize(fun=modFreeEnergy, x0=a, method="Nelder-Mead",callback=afterEachIteration,options={'disp': True})

# POSSIBLE ARGUMENTS
#scipy.optimize.minimize(fun, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})

print(res)

finalSolution = res.x.reshape((nx,ny,4))

# saving solution in Python format, get back i.e. by a = np.load("finalSolution.npy")
np.save('finalSolution', finalSolution)

# saving human readable

aR = finalSolution[:,:,0]
aI = finalSolution[:,:,1]
bR = finalSolution[:,:,2]
bI = finalSolution[:,:,3]

np.savetxt('finalSolution-aR.txt', aR)
np.savetxt('finalSolution-aI.txt', aI)
np.savetxt('finalSolution-bR.txt', bR)
np.savetxt('finalSolution-bI.txt', bI)

print("final solution")
#print(finalSolution)
print("with free energy")
print(modFreeEnergy(finalSolution))
print("--- %s seconds ---" % (time.time() - start_time))


######## PLOTTING RESULTS ###############
dx = DX(finalSolution)
np.save('finalSolution-dx', dx)
np.savetxt('finalSolution-dx-aR.txt', dx[:,:,0])
np.savetxt('finalSolution-dx-aI.txt', dx[:,:,1])
np.savetxt('finalSolution-dx-bR.txt', dx[:,:,2])
np.savetxt('finalSolution-dx-bI.txt', dx[:,:,3])

dy = DY(finalSolution)
np.save('finalSolution-dy', dy)
np.savetxt('finalSolution-dy-aR.txt', dy[:,:,0])
np.savetxt('finalSolution-dy-aI.txt', dy[:,:,1])
np.savetxt('finalSolution-dy-bR.txt', dy[:,:,2])
np.savetxt('finalSolution-dy-bI.txt', dy[:,:,3])




plot(finalSolution,["Re(Deltax)","Im(Deltax)","Re(Deltay)","Im(Deltay)"] )
#plot(np.gradient(finalSolution[:,:,:], axis=0)/hx,["BI dx Re(Deltax)"," dx Im(Deltax)"," dx Re(Deltay)","dx Im(Deltay)"] )
plot(dx,["dx Re(Deltax)","dx Im(Deltax)","dx Re(Deltay)","dx Im(Deltay)"] )
#plot(np.gradient(finalSolution[:,:,:], axis=1)/hy,["BI dy Re(Deltax)"," dy Im(Deltax)"," dy Re(Deltay)","dy Im(Deltay)"] )
plot(dy,["dy Re(Deltax)","dy Im(Deltax)","dy Re(Deltay)","dy Im(Deltay)"] )

plt.show()

