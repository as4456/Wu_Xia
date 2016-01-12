import numpy as np

def TAU_P(X0,mu,rho,sigma,delta0,rlb,N):
    

    tau = np.empty([N, 1]); tau[:] = np.NAN
    S0 = delta0 + X0[0] + X0[1]
    if S0 >= rlb:
        raise Exception('It is above the lower bound at t=0')
    
    for n in range(0, N):
        X_temp = X0
        X_temp = np.reshape(X_temp, (X_temp.shape[0], 1))
        S_temp = delta0 + X_temp[0] + X_temp[1]
        counter = 0
        while(S_temp < rlb):
            X_temp = mu[:, None] + np.dot(rho,X_temp) + np.dot(sigma, np.random.randn(len(X0),1))
            counter = counter + 1
            S_temp = delta0 + X_temp[0] + X_temp[1]
        tau[n] = counter
    return(tau)
        
