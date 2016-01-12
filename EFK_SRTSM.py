import numpy as np
from scipy.stats import norm

def EFK_SRTSM(parameters, maturities, forwardrates, index):
    
    forwardrates = forwardrates.T
    J = maturities
    T = forwardrates.shape[1]
    #Parameterization
    rhoP = parameters[0:9];
    rhoP = np.reshape(rhoP, [3,3], order='F')
    muP = parameters[9:12]
    rhoQ1 = parameters[12]; rhoQ2 = parameters[13];
    sigma = [[abs(parameters[14]), 0, 0],
        [parameters[15], abs(parameters[17]), 0],
        [parameters[16], parameters[18], abs(parameters[19])]]
    omega = np.dot(sigma, np.transpose(sigma))
    sigma11 = omega[0,0]; sigma22 = omega[1,1]; sigma33 = omega[2,2]; sigma12 = omega[0,1]; sigma13 = omega[0,2]; sigma23 = omega[1,2];
    delta0 = parameters[20]
    omegaM = np.zeros((len(J), len(J))); np.fill_diagonal(omegaM, parameters[21]**2)
    rlb = parameters[22]
    
    
    # Initialization
    zt = np.empty([len(J), 6])
    zt[:] = np.NAN
    
    xt = rhoQ1**2
    if abs(xt-1) > 1e-5:
        zt[:,0] = (xt**J-1)/(xt-1)
    else:
        zt[:,0] = J
        
    xt = rhoQ2**2
    if abs(xt-1) > 1e-5:
        zt[:,1] = (xt**J-1)/(xt-1)
    else:
        zt[:,1] = J
        
    xt = rhoQ2**2
    if abs(xt-1) > 1e-5:
        zt[:,2] = xt**J * ((J**2*xt**2)-(2*J**2*xt)+J**2-2*J*xt**2+2*J*xt+xt**2+xt)/(xt-1)**3-xt*(xt+1)/(xt-1)**3;
    else:
        zt[:,2] = (J-1)*J*(2*J-1)/6
        
    xt = rhoQ1*rhoQ2
    if abs(xt-1) > 1e-5:
        zt[:,3] = (xt**J-1)/(xt-1)
    else:
        zt[:,3] = J
        
    xt = rhoQ1*rhoQ2
    if abs(xt-1) > 1e-5:
        zt[:,4] = xt/(xt-1)**2 - xt**J*(J+xt-J*xt)/(xt-1)**2
    else:
        zt[:,4] = J*(J-1)/2
        
    xt = rhoQ2**2
    if abs(xt-1) > 1e-5:
        zt[:,5] = xt/(xt-1)**2 - xt**J*(J+xt-J*xt)/(xt-1)**2
    else:
        zt[:,5]  = J*(J-1)/2
    
    temp = np.array([sigma11, sigma22, (1/rhoQ2**2)*sigma33, 2*sigma12, 2/rhoQ2*sigma13, 2/rhoQ2*sigma23])
    sigmasJ2 = np.dot(zt, temp)
    del temp
    
    sigmasJ = np.sqrt(sigmasJ2)
    JJ = np.arange(1, max(J)+1, 1)
    JJ = np.transpose(JJ)
    
    bn = np.column_stack((rhoQ1**JJ, rhoQ2**JJ, JJ*rhoQ2**(JJ-1)))
    Bn = np.vstack((np.array([1,1,0]), bn))
    Bn = np.cumsum(Bn, axis = 0)
    Bn = Bn[0:-1,:]
    
    aJ = np.empty([len(J),1])
    aJ[:] = np.NAN
    
    for j in range(0, len(J)):
        aJ[j] = delta0 - 0.5*np.dot(Bn[J[j]-1,:], np.dot(np.dot(sigma, np.transpose(sigma)), np.transpose(Bn[J[j]-1,:])))/1200

    bJ = np.column_stack((rhoQ1**J, rhoQ2**J, J*rhoQ2**(J-1)))
    
    ## Filtering the shadow rate
    # Initialization
    X1 = np.empty([3,T+1]); X1[:] = np.NAN; X1[:,0] = 0 # E_t-1[X_t]; t = 1,...,T+1
    V1 = np.empty([3,3,T+1]); V1[:] = np.NAN; V1[:,:,0] = np.eye(3, dtype=int)*100
    V1[2,2,0] = V1[2,2,0]/144
    loglikvec = np.empty([1,T]); loglikvec[:] = np.NAN
    X2 = np.empty([3,T+1]); X2[:] = np.NAN
    V2 = np.empty([3,3,T+1]); V2[:] = np.NAN
    for t in range(0, T):
        musJ = aJ + np.dot(bJ, np.reshape(X1[:,t], [3,1]))
        z1_temp = (musJ-rlb)/sigmasJ.reshape((sigmasJ.shape[0],1))
        z2_temp = rlb + (musJ-rlb)*norm.cdf(z1_temp) + sigmasJ.reshape((sigmasJ.shape[0],1))*norm.pdf(z1_temp);
        H = norm.cdf(z1_temp)
        H = np.column_stack((H, H,H))*bJ
        err = np.reshape(forwardrates[:,t], [forwardrates.shape[0],1]) - z2_temp
        S = np.dot(H, np.dot(V1[:,:,t],np.transpose(H))) + omegaM
        invS = np.linalg.inv(S)
        K = np.dot(np.dot(V1[:,:,t], np.transpose(H)), invS);
        
        # Calculate log liklihood
        loglikvec[0,t] = len(J)*np.log(2*np.pi) + np.log(np.linalg.det(S)) + np.dot(np.dot(np.transpose(err), invS), err)
        loglikvec[0,t] = (-0.5)*loglikvec[0,t]
        
        # Update
        X2[:,t+1] = X1[:,t] + np.reshape(np.dot(K,err), np.shape(K)[0]);
        V2[:,:,t+1] = np.dot((np.eye(3)-np.dot(K,H)),V1[:,:,t]);
        
        # Predict
        X1[:,t+1] = muP + np.dot(rhoP,X2[:,t+1]);
        V1[:,:,t+1] = np.dot(np.dot(rhoP,V2[:,:,t+1]),np.transpose(rhoP)) + omega;
    
    llf = np.sum(loglikvec)
    SR = delta0 + X2[0,1:] + X2[1,1:] 
    Xf = X2[:,1:] 
    
    if index == 0:
        y = -llf;
    elif index ==1:
        y = SR; 
    elif index ==2:
        y = Xf; 
    else:
        F_shadow = np.tile(aJ,(1,T)) + np.dot(bJ,Xf); 
        z1_temp = (F_shadow - rlb)/np.tile(sigmasJ[:,None],(1,T)); 
        y = rlb + (F_shadow - rlb)*norm.cdf(z1_temp) + np.tile(sigmasJ[:,None],(1,T))*norm.pdf(z1_temp);
        
    return y