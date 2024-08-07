import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


################## neuronal culture ##################
def create_pdms(**kwargs):
    """
    simulate the petri dish and the pdms barriers for neuronal culture.
    """
    
    params = {
        # network (adjacency matrix) parameters
        'h'        : 100,  # grid's pixels (the shape of pdms is in pixel units)
        'width'    : 3,    # culture size mm
        'heigth'   : 3,    # culture size mm
        'Depth'    : 0.1,  # pdms barrier height (average)
        'PDMS'     : 'Flat'# kind of pdms support 
    }
    params.update(kwargs)

    # Extract parameters
    h = params['h']
    Depth = params['Depth']
    PDMS = params['PDMS']
    
    # Create matrix
    H = np.zeros((h, h))
    
    # Set radius, center
    radius, center = h // 2.001, (h - 1) / 2
    
    # simulate PDMS barriers
    tracks_dpt = []
    if PDMS == 'Tracks': 
        # Fill the matrix
        for i in range(h):
            for j in range(h):
                if (i - center) ** 2 + (j - center) ** 2 >= radius ** 2:
                    H[i, j] = -1
        
        # Fill alternate columns
        for i in range(h):
            if i % 10 == 0:
                trk_dpt = Depth + np.random.normal(0, Depth/50)
                tracks_dpt.append(trk_dpt)
                H[:, i:i+5] = trk_dpt
                
    # Set values of pdms to -1 if outside the sphere (infinite barriers)
    # if PDMS = 'Flat' just do a circular barrier
    for j in range(h):
        for i in range(h):
            if (i - center) ** 2 + (j - center) ** 2 >= radius ** 2:
                H[i, j] = -1
    
    return H, tracks_dpt, h


def place_neurons(width, height, H=[], **kwargs):
    """
    randomly place neurons on a dish defined by H
    """
    
    width       = width     # culture width (mm)
    height      = height    # culture height (mm)
    H           = H         # obstacles MxN grid of numbers:
                            #   0   : no obstacle; 
                            #   h>0 : obstacle of height h mm; 
                            #   h<0 : no neurons or axons can grow here

    # default parameters:
    rho     = 100       # neuron density (neurons / mm^2)
    r_soma  = 7.5e-3    # soma radius (mm)
    
    for argn, argv in kwargs.items():
        match argn:
            case 'rho':
                rho = float(argv)
            case 'r_soma':
                r_soma = float(argv)

    # some derived parameters:
    M = int(height*width*rho)                          # number of neurons  
    Hm,Hn = np.shape(H) if np.size(H) > 0 else (1,1)   # size of obstacles (PDMS) grid
    cell_height = height/Hm
    cell_width = width/Hn
    
    # place neurons non overlapping on area:
    X,Y = np.zeros((2,M))
    X[0] = np.random.rand()*width
    Y[0] = np.random.rand()*height
    for i in range(1,M):
        X[i] = np.random.rand()*width
        Y[i] = np.random.rand()*height
        r,c = get_cell(X[i], Y[i], cell_width, cell_height)
        while np.any(np.sqrt(np.power(X[:i]-X[i],2)+np.power(Y[:i]-Y[i],2)) < r_soma) or get_H(r,c,H) < 0:
            X[i] = np.random.rand()*width
            Y[i] = np.random.rand()*height
            r,c = get_cell(X[i], Y[i], cell_width, cell_height)

    return X,Y

def grow_W(width, height, X, Y, H=[], keep_ax = False, **kwargs):
    """
    grow axons and establish connections between neurons.
    see Orlandi et al. 2013 "Noise focusing and the emergence of coherent activity in neuronal cultures"
    """
    width       = width     # culture width (mm)
    height      = height    # culture height (mm)
    X           = X         # array X coordinates 
    Y           = Y         # array Y coordinates
    H           = H         # obstacles MxN grid of numbers:
                            #   0   : no obstacle; 
                            #   h>0 : obstacle of height h mm; 
                            #   h<0 : no neurons or axons can grow here

    # default arguments:
    L       = 1.0   # average axon length (mm)
    Dl      = 1e-3  # axon segment length (mm)
    phi_sd  = 0.1   # axon random walk std

    r_dendrite_mu   = 150e-3    # denrite radius mean (mm)
    r_dendrite_sd   = 20e-3     # denrite radius std (mm)

    # read optional arguments:
    for argn, argv in kwargs.items():
        match argn:
            case 'L':
                L = float(argv)
            case 'Dl':
                Dl = float(argv)
            case 'phi_sd':
                phi_sd = float(argv)
            case 'r_dendrite_mu':
                r_dendrite_mu = float(argv)
            case 'r_dendrite_sd':
                r_dendrite_sd = float(argv)

    # some derived parameters:
    M = len(X)                 # number of neurons
    Hm,Hn = np.shape(H) if np.size(H) > 0 else (1,1)  # size of obstacles (PDMS) grid
    cell_height = height/Hm    # the height (in mm) of each "pixel" (each unit of matrix H)
    cell_width = width/Hn

    # init:
    W = np.zeros((M,M)) # resulting adjacency matrix

    r_dendrite = r_dendrite_mu + r_dendrite_sd*np.random.normal(0,1,M) # dendritic tree radius for all neurons

    Ln = L*np.sqrt(-2*np.log(1-np.random.rand(M)))  # array of axon lengths  
    Nl = int(np.max(Ln)/Dl)                         # number of segments of longest axon

    Xi = np.copy(X)                                 # array: X coordinates of all neurons
    Yi = np.copy(Y)                                 # array: Y coordinates of all neurons
    phi = 2*np.pi*np.random.rand(M)                 # array: initial angles of first segments 

    # keep track of axons
    if keep_ax == True:
        X_a = np.zeros((M,np.max(Nl)),dtype = 'float')
        Y_a = np.zeros((M,np.max(Nl)),dtype = 'float')

        X_a[:,0] = X
        Y_a[:,0] = Y

    # main loop:
    for n in tqdm(np.arange(1,Nl)): # run over the max number of segments 
        # determine axon growth direction:
        Dx = np.multiply(Dl*np.cos(phi), (n*Dl)<Ln)
        Dy = np.multiply(Dl*np.sin(phi), (n*Dl)<Ln)

        # temporarily update axon cone:
        Xj = Xi+Dx
        Yj = Yi+Dy

        for i in np.arange(M)[np.array((n*Dl)<Ln)]:
            # check for border crossings:
            # it returns 1) the position of the final point, 2) the angle in case of deflection and 3) the total final lenght
            _, Xj[i], Yj[i], phi[i], Ln[i] = check_crossing(Xi[i], Yi[i], Xj[i], Yj[i], H, phi[i], Ln[i], Dl, cell_width, cell_height)
            
            # determine connections:
            P = np.sqrt(np.power(Xj[i]-X,2) + np.power(Yj[i]-Y,2)) < r_dendrite 
            P[i] = 0    # no self-connections
            for j in np.arange(M)[P==1]:    # for all neurons for which axon is close
                # check whether dendrites cross border and a connection is established :
                crossed, _, _, _, _ = check_crossing(X[j], Y[j], Xj[i], Yj[i], H, phi[i], Ln[i], Dl, cell_width, cell_height)
                if crossed:
                    W[i,j] = 1

        # update axon cone positions:
        Xi = Xj
        Yi = Yj

        # random deviation for next step:
        phi += phi_sd*np.random.normal(0, 1, M)

        if keep_ax == True:
            X_a[:,n] = Xj
            Y_a[:,n] = Yj 

    if keep_ax == True:
        return W, X_a, Y_a
        
    else:
        return W

def get_cell(x, y, cell_width, cell_height):
    """
    measures in standard units the coordinates x and y
    """
    row = int(y/cell_height)
    col = int(x/cell_width)
    return row,col

def get_H(row, col, H):
    """
    return the a position within the grid H, in grid units (pythonic: from 0 to L-1)
    """
    if len(H) == 0: return 0

    M,N = np.shape(H)
    
    row = 0 if row < 0 else ( M-1 if row >= M else row ) 
    col = 0 if col < 0 else ( N-1 if col >= N else col )

    return H[row,col]

def get_P(H_i, H_j):
    # values from Hernandez-Navarro thesis
    h = np.array([ 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p01 = np.array([ 1, 0.0007, 0.00045, 0.00035, 0.00030, 0.00025, 0.00015, 0.00002, 0 ])
    p10 = np.array([ 1, 0.0066, 0.0033, 0.0033, 0.0033, 0.0033, 0.00200, 0.00050, 0 ]) 
    Dh = H_j-H_i
    if Dh > 0:
        return np.interp(Dh, h, p01)
    else:
        return np.interp(abs(Dh), h, p10)

def check_crossing(xi, yi, xj, yj, H, phi, Ln, Dl, cell_width, cell_height):
    crossed = True
    if len(H) > 0:
        re_check = True
        while re_check:
            re_check = False
            row_i, col_i = get_cell(xi, yi, cell_width, cell_height) # return the position of point (xi, yi) on the grid H, in grid/list units
            row_j, col_j = get_cell(xj, yj, cell_width, cell_height)

            H_i = get_H(row_i, col_i, H)    # in which point of the grid is the point i
            H_j = get_H(row_j, col_j, H)    # in which point of the grid is the point j

            P_cross = get_P(H_i, H_j) # the probability an axon cross a barrier
            if P_cross < 1: # so, maybe we do not cross

                # calculate coordinates of the crossed wall (in mm) 
                xborder = -1 if col_i == col_j else np.max([col_i, col_j])*cell_width 
                yborder = -1 if row_i == row_j else np.max([row_i, row_j])*cell_height

                A = (xi, yi)
                B = (xj, yj)
                if xborder >= 0 and yborder >= 0:
                    # we are crossing 2 borders, life is hard
                    Px = (xborder, -1000*cell_height)
                    Qx = (xborder, 1000*cell_height)
                    Py = (-1000*cell_width, yborder)
                    Qy = (1000*cell_width, yborder)
                    P,Q = (Py, Qy) if distance(A, Py, Qy) < distance(A, Px, Qx) else (Px, Qx)
                elif xborder >= 0:
                    P = (xborder, -1000*cell_height)
                    Q = (xborder, 1000*cell_height)
                elif yborder >= 0:
                    P = (-1000*cell_width, yborder)
                    Q = (1000*cell_width, yborder)
                else: pass  # should not happen though

                angle_AB_on_PQ = get_angle(A, B, P, Q)
                if ( abs(angle_AB_on_PQ) <= np.pi/6.0 ) or ( np.random.rand() < (1-P_cross) ):
                    # deflect axon:
                    phi = get_slope(P, Q) if angle_AB_on_PQ < 0 else get_slope(Q, P)
                    xj = xi + Dl*np.cos(phi)
                    yj = yi + Dl*np.sin(phi)
                    re_check = True
                    crossed = False
                else:
                    # axon crosses the obstacle: the egight of the barrier is subtracted (the axons are thought horizontal)
                    Ln -= abs(H_j-H_i)
    return crossed, xj, yj, phi, Ln

def distance(A,P,Q):
    num = (Q[0]-P[0])*(P[1]-A[1]) - (P[0]-A[0])*(Q[1]-P[1])
    den = np.sqrt((Q[0]-P[0])**2+(Q[1]-P[1])**2)
    return num/den

def get_slope(A,B):
    return np.arctan2(B[1]-A[1],B[0]-A[0])

def get_angle(A,B,P,Q):
    ''' angle of AB w. resp of PQ '''
    return np.angle(np.exp(1j*(get_slope(A,B)-get_slope(P,Q))))

##################### NEURONAL DYNAMICS ########################

"""
the dynamics of the neuronal culture is simulated following the Izhikevich’s integrate-and-fire model:
see Izhikevich (2003) "Simple model of spiking neurons"

The addition of a pre-synaptic depression dynamics accounts for neurotransmitter depletion following repetitive firing: 
see Alvarez-Lacalle, and Moses (2009) "Slow and fast pulses in 1-D cultures of excitatory neurons"
"""

def run_nw(nw):
    Nt = nw['params']['Nt']
    for n in tqdm(np.arange(1,Nt)):
        simulation_step(n,nw)


def simulation_step(n, nw):

    v = nw['v']
    u = nw['u']
    P = nw['P']
    R = nw['R']
    dv = nw['dv']
    du = nw['du']
    dP = nw['dP']
    dR = nw['dR']
    Inp = nw['Inp']
    eta = nw['eta']

    pars = nw['params']

    Ne  = nw['Ne']
    Dt  = pars['Dt']

    Inp[:]  = np.dot(nw['W'], P)                         # synaptic input
    eta[:]  = pars['gW']*np.random.normal(0,1,nw['N'])   # noise drive

    du[:]   = pars['a'] * (pars['b'] * v - u)
    dP[:Ne] = -P[:Ne]/pars['tauE']
    dP[Ne:] = -P[Ne:]/pars['tauI']
    dR[:]   = (1-R)/pars['tauR']

    dv[:]   = pars['alpha']*np.power(v,2) + pars['beta']*v + pars['gamma'] - u
    v[:]    = v + 0.5*(dv + Inp + eta)*Dt
    dv[:]   = pars['alpha']*np.power(v,2) + pars['beta']*v + pars['gamma'] - u
    v[:]    = v + 0.5*(dv + Inp + eta)*Dt

    u[:]    = u + du*Dt
    P[:]    = P + dP*Dt
    R[:]    = R + dR*Dt

    for i in np.arange(len(v))[v>pars['vc']]:
        v[i] = pars['v0'][i]
        u[i] = u[i] + pars['delta_u'][i]
        P[i] = P[i] + R[i]
        R[i] = (1-pars['beta_R'])*R[i]
        nw['spike_T'].append(n*Dt)
        nw['spike_Y'].append(i)

    return nw   # do not capture, nw is updated by reference



def init_network(N, A, **kwargs):
    params = {
        'P_connect': 0.5,        # percentage of retained links (bigger = more final links)
        'EI_ratio':  0.8,
        'gE':        6,          # excitatory strength
        'gI':        -12,        # inhibitory strength
        'Inhibition':True,       # in case of EI_ratio = 0/1 (only Inh/Exc neurons), consider inhibitions in the adj matrix

        # distribution of neurons by class of dynamics:
        
        # excitatory
        'neu_type_exc': {
            'RS': 100, # regular spiking
            'IB': 0,  # intrinsically burtsting
            'CH': 0,  # chattering
    
            # cortex input neurons
            'TC': 0, # thalamo-cortical
    
            # others
            'RZ': 0, # resonator 
        },
        
        # inhibitory
        'neu_type_inh': {                
            'FS': 100, # fast spiking
            'LTS': 0, # low tresholding spiking
        },

        # Izhikevich parameters:
        'alpha': 0.04,
        'beta': 5,
        'gamma': 140,

        'a_0' : [0.02, 0.1],
        'b_0' : [0.2, 0.25],
        'v0_0': [-65, -55, -50],
        'delta_u_0' : [0.05, 2, 4, 8],

        'Izhikevich_par_for_reference': {'Exc': ['RS', 'IB', 'CH', 'TC', 'RZ'], 
                 'Inh': ['FS', 'LTS'],
                 'RS': ['a = 0.02, b = 0.2, v0 = -65, delta_u = 8'],
                 'IB': ['a = 0.02, b = 0.2, v0 = -55, delta_u = 4'],
                 'CH': ['a = 0.02, b = 0.2, v0 = -50, delta_u = 2'],
                 'TC': ['a = 0.02, b = 0.25, v0 = -65, delta_u = 0.05'],
                 'RZ': ['a = 0.1, b = 0.26, v0 = -65, delta_u = 2'],
                 'FS': ['a = 0.1, b = 0.2, v0 = -65, delta_u = 2'],
                 'LTS': ['a = 0.02, b = 0.25, v0 = -65, delta_u = 2'],
                },
        
        'noise_param': 0.08, # noise for parameter values, in standard deviations units
        
        'vc': 30,   # threshold

        # synapse dynamics:
        'tauE': 10,   # exc. time-scale (decay)
        'tauI': 10,   # inh. time-scale (decay)
        'beta_R': 0.8,  # R-`reset' (mult. factor)
        'tauR': 8e3,  # synaptic depression recovery time-scale

        # noise driving:
        'sigma': 2.25,   # noise magnitude

        # sim. parameters:
        'Dt': 1e0,  # time-step size (ms)
        'Tn': 1e3,  # total sim. time (ms)
    }

    # Update params:
    params.update(kwargs)

    # neuronal-type-distribution (as percentage of N)
    pe = [params['neu_type_exc']['RS'], params['neu_type_exc']['IB'], params['neu_type_exc']['CH'], 
          params['neu_type_exc']['TC'], params['neu_type_exc']['RZ']]                              # excitatory
    pi = [params['neu_type_inh']['LTS'], params['neu_type_inh']['FS']]                             # inhibitory

    # partition the neurons by type
    EI_partitions = partition_set(N, params['EI_ratio'], pe, pi)

    # I/E proportion
    Ne = np.sum(EI_partitions['E_partition']) # number of excitatory neurons
    Ni = np.sum(EI_partitions['I_partition']) # number of inhibitory neurons
    
    if Ni == N or Ne == N:
        if params['Inhibition'] == True:
            Ne = int(N * params['EI_ratio'])
            Ni = N - Ne

        
    # Izhikevich final parameters
    de = EI_partitions['E_partition']
    di = EI_partitions['I_partition']
    # parameter a
    ae = np.hstack([params['a_0'][0] * np.random.normal(1,params['noise_param'],de[0]), # 'RS'
                    params['a_0'][0] * np.random.normal(1,params['noise_param'],de[1]), # 'IB'
                    params['a_0'][0] * np.random.normal(1,params['noise_param'],de[2]), # 'CH'
                    params['a_0'][0] * np.random.normal(1,params['noise_param'],de[3]), # 'TC'
                    params['a_0'][1] * np.random.normal(1,params['noise_param'],de[4])])# 'RZ'
    ai = np.hstack([params['a_0'][0] * np.random.normal(1,params['noise_param'],di[0]), # 'LTS'
                    params['a_0'][1] * np.random.normal(1,params['noise_param'],di[1])])# 'FS' 

    # parameter b
    be = np.hstack([params['b_0'][0] * np.random.normal(1,params['noise_param'],de[0]), # 'RS'
                    params['b_0'][0] * np.random.normal(1,params['noise_param'],de[1]), # 'IB'
                    params['b_0'][0] * np.random.normal(1,params['noise_param'],de[2]), # 'CH'
                    params['b_0'][1] * np.random.normal(1,params['noise_param'],de[3]), # 'TC'
                    params['b_0'][1] * np.random.normal(1,params['noise_param'],de[4])])# 'RZ'
    bi = np.hstack([params['b_0'][0] * np.random.normal(1,params['noise_param'],di[0]), # 'LTS'
                    params['b_0'][1] * np.random.normal(1,params['noise_param'],di[1])])# 'FS' 

    # parameter v0 (aka 'c')
    v0e = np.hstack([params['v0_0'][0] * np.random.normal(1,params['noise_param'],de[0]), # 'RS'
                     params['v0_0'][1] * np.random.normal(1,params['noise_param'],de[1]), # 'IB'
                     params['v0_0'][2] * np.random.normal(1,params['noise_param'],de[2]), # 'CH'
                     params['v0_0'][0] * np.random.normal(1,params['noise_param'],de[3]), # 'TC'
                     params['v0_0'][0] * np.random.normal(1,params['noise_param'],de[4])])# 'RZ'
    v0i = np.hstack([params['v0_0'][0] * np.random.normal(1,params['noise_param'],di[0]), # 'LTS'
                     params['v0_0'][0] * np.random.normal(1,params['noise_param'],di[1])])# 'FS'

    # parameter delta_u (aka 'd')
    delta_ue = np.hstack([params['delta_u_0'][3] * np.random.normal(1,params['noise_param'],de[0]), # 'RS'
                          params['delta_u_0'][2] * np.random.normal(1,params['noise_param'],de[1]), # 'IB'
                          params['delta_u_0'][1] * np.random.normal(1,params['noise_param'],de[2]), # 'CH'
                          params['delta_u_0'][0] * np.random.normal(1,params['noise_param'],de[3]), # 'TC'
                          params['delta_u_0'][1] * np.random.normal(1,params['noise_param'],de[4])])# 'RZ'
    delta_ui = np.hstack([params['delta_u_0'][1] * np.random.normal(1,params['noise_param'],di[0]), # 'LTS'
                          params['delta_u_0'][1] * np.random.normal(1,params['noise_param'],di[1])])# 'FS'

    # final vectors of parameters for all neurons
    params['a'] = np.concatenate((ae, ai))
    params['b'] = np.concatenate((be, bi))
    params['v0'] = np.concatenate((v0e, v0i))
    params['delta_u'] = np.concatenate((delta_ue, delta_ui))

    params['Nt'] = int(params['Tn'] / params['Dt'])
    params['gW'] = np.sqrt(2 * params['sigma'] / params['Dt'])

    # Set seed:
    if 'seed' in params: np.random.seed(params['seed'])

    # define adjacency matrix
    B = np.multiply(A, np.random.rand(N, N) > (1 - params['P_connect']))                         # cancel connections for sparsity
    
    # set weights
    W = np.multiply(B, params['gE'] * np.abs(np.random.normal(1, 0.1, (N, N))))                  # set excitatory weights
    W[Ne:, :] = np.multiply(B[Ne:, :], params['gI'] * np.abs(np.random.normal(1, 0.1, (Ni, N)))) # set inhibitory weights
        

        # # set weights - alternative 
        # W = np.multiply(B, params['gE'] * np.random.rand(N, N))                  # set excitatory weights 
        # W[Ne:, :] = np.multiply(B[Ne:, :], params['gI'] * np.random.rand(Ni, N))  # set inhibitory weights

    
    v = np.multiply(np.random.rand(N), params['vc']) + params['v0']
    u = np.multiply(params['b'], v)

    P = np.random.rand(N)
    R = np.random.rand(N)


    nw = {
        'N': N,
        'Ne': Ne,
        'Ni': Ni,

        'A': A,
        'W': W,

        'v': v,
        'u': u,
        'P': P,
        'R': R,

        'dv': v * 0,
        'du': u * 0,
        'dP': P * 0,
        'dR': R * 0,

        'Inp': v * 0,
        'eta': v * 0,

        'spike_T': [],
        'spike_Y': [],

        'params': params,

        # include neuron types partition in the dictionsry, to label neurons
        'EI_partitions': EI_partitions,

        # Include a, b, v0, and delta_u in the output dictionary
        'a': params['a'],
        'b': params['b'],
        'v0': params['v0'],
        'delta_u': params['delta_u'],
    }
    
    return nw

#######################


def create_firing_matrix(spk_ids, spk_time, T, N):
    """
    Create a firing matrix of size N*T from spike IDs and spike times.

    Parameters:
    - spk_ids: List or array of neuron IDs that spiked.
    - spk_time: List or array of spike times corresponding to spk_ids.
    - T: Total time steps.
    - N: Total number of neurons.

    Returns:
    - firing_matrix: A binary matrix of size N*T where each element (i, j) is 1
      if neuron i fired at time j, and 0 otherwise.
    """
    # Create the firing matrix filled with zeros
    firing_matrix = np.zeros((N, T), dtype=int)
    
    # Create a mapping from neuron ID to its index in the firing matrix
    unique_ids = np.unique(spk_ids)
    id_to_index = {id: index for index, id in enumerate(unique_ids) if id < N}
    
    # Populate the firing matrix
    for id, time in zip(spk_ids, spk_time):
        if id in id_to_index and 0 <= int(time) < T:
            firing_matrix[id_to_index[id], int(time)] = 1
    
    return firing_matrix



def partition_set(N, EI_ratio, ve, vi):
    '''
    in:
      ve/vi are lists of proportions of neuronal type, repsectively for excitatory and inhibitory neurons, for example
      ve = [50, 30, 0, 20, 0]
      vi = [100, 0]

    out: 
      two lists representing the number of neurons for each type. 
    '''
    if sum(ve) > 0 and sum(ve) != 100:
        raise ValueError("ve must sum to 100")
    if sum(vi) > 0 and sum(vi) != 100:
        raise ValueError("vi must sum to 100")

    if sum(vi) == 0:
        EI_ratio = 1
    elif sum(ve) == 0:
        EI_ratio = 0

    # Convert percentages to proportions
    E_proportion = EI_ratio
    I_proportion = 1 - EI_ratio

    # Calculate the number of elements in each main group (Excitatory vs Inhibitory)
    Ne = int(round(E_proportion * N))
    Ni = N - Ne  # Ensure the total is N

    # Calculate the number of elements in each subgroup of the first main group
    Ne_subgroups = [int(round(v * Ne / 100)) for v in ve]
    Ne_adjustment = Ne - sum(Ne_subgroups)
    if Ne_adjustment != 0:
        # Adjust the largest subgroup in Ne to ensure the sum is correct
        largest_index = np.argmax(Ne_subgroups)
        Ne_subgroups[largest_index] += Ne_adjustment

    # Calculate the number of elements in each subgroup of the second main group
    Ni_subgroups = [int(round(v * Ni / 100)) for v in vi]
    Ni_adjustment = Ni - sum(Ni_subgroups)
    if Ni_adjustment != 0:
        # Adjust the largest subgroup in Ni to ensure the sum is correct
        largest_index = np.argmax(Ni_subgroups)
        Ni_subgroups[largest_index] += Ni_adjustment

    return {
        'E_partition': Ne_subgroups,
        'I_partition': Ni_subgroups
    }

###############################
    
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_neurons_and_pdms(X, Y, H, width, height, h):
    """
    Plots neuron positions overlaid on PDMS support.

    Parameters:
    - X: List or array of x coordinates of neurons.
    - Y: List or array of y coordinates of neurons.
    - H: 2D array representing the PDMS support image.
    - width: Width of the culture in mm.
    - height: Height of the culture in mm.
    - h: Size parameter for scaling tick labels.
    """
    fig = plt.figure(figsize=(7, 3))  # Define figure size

    # PDMS support subplot
    ax = plt.subplot(111)
    im = ax.imshow(H, origin='lower')
    ax.set_title('PDMS support with Neuron Positions', fontsize=12)  # Set title font size
    ax.set_xlabel('culture width (mm)')
    
    # Add colorbar with legend
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PDMS depth', fontsize=12)

    # Change the tick labels
    ax.set_xticks(np.arange(0, h+1, 20))  # Set tick positions
    ax.set_yticks(np.arange(0, h+1, 20))

    # Scale the tick labels
    # Define the conversion factor from matrix indices to mm
    conversion_x = width / h  # width mm corresponds to h
    conversion_y = height / h  # height mm corresponds to h
    x_tick_labels = [f'{val * conversion_x:.2f}' for val in np.arange(0, h+1, 20)]
    y_tick_labels = [f'{val * conversion_y:.2f}' for val in np.arange(0, h+1, 20)]
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    # Normalize X, Y coordinates to fit the scale of the PDMS support plot
    X_norm = np.array(X) * (h / width)
    Y_norm = np.array(Y) * (h / height)

    # Overlay the neuron positions on the PDMS support subplot
    ax.plot(X_norm, Y_norm, 'o', markersize=1, color='red', label='Neuron Positions')

    plt.tight_layout()  # Adjust layout to prevent overlap
    
    plt.show()







