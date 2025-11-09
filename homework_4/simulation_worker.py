import numpy as np
from numba import njit

@njit
def acceleration(rvec,vvec,A,B):
    mag_r = np.sqrt(rvec[0]**2 + rvec[1]**2)
    mag_v = np.sqrt(vvec[0]**2 + vvec[1]**2)
    if mag_r == 0.0:
        term1 = np.array([0.0, 0.0])
    else:
        term1 = (-1.0 / (4.0 * (mag_r**3))) * rvec
    if mag_v == 0.0:
        term2 = np.array([0.0, 0.0])
    else:
        term2 = (A / (mag_v**3 + B)) * vvec
    return term1 - term2

@njit                                             
def _rk4_step_numba(h_step, rvec, vvec, A, B):
    k1_r = h_step * vvec
    k1_v = h_step * acceleration(rvec, vvec, A, B)

    r_mid1 = rvec + 0.5 * k1_r
    v_mid1 = vvec + 0.5 * k1_v

    k2_r = h_step * v_mid1
    k2_v = h_step * acceleration(r_mid1, v_mid1, A, B)

    r_mid2 = rvec + 0.5 * k2_r
    v_mid2 = vvec + 0.5 * k2_v

    k3_r = h_step * v_mid2
    k3_v = h_step * acceleration(r_mid2, v_mid2, A, B)
        
    r_end = rvec + k3_r
    v_end = vvec + k3_v
    k4_r = h_step * v_end
    k4_v = h_step * acceleration(r_end, v_end, A, B)
        
    r_new = rvec + (1/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = vvec + (1/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
    return r_new, v_new

def rk4_integrate(A, B, h, r_init, v_init, delta,**kwargs):

    pos_vec = r_init                        # though not written, the initial positions are x=1, y=0
    vel_vec = v_init                        # initial velocity is completely in the y
    t = 0
    h_step = h

    pos_list = []
    vel_list = []
    t_list = []

    pos_list.append(pos_vec)
    vel_list.append(vel_vec)
    t_list.append(t)

    min_error = 1e-20

    if 't_final' in kwargs:
        t_final = kwargs['t_final']

        while (t < t_final):

            r_mid, v_mid = _rk4_step_numba(h_step, pos_vec, vel_vec, A, B)
            rvec_1, vvec_1 = _rk4_step_numba(h_step, r_mid, v_mid, A, B)
            rvec_2, vvec_2 = _rk4_step_numba(2*h_step, pos_vec, vel_vec, A, B)

            error = np.linalg.norm(rvec_2 - rvec_1) + min_error
            rho = (30 * h_step * delta)/error

            if rho >= 1:
                t += 2*h_step                                       # accept current h_step
                pos_vec = rvec_1 + (1/15)*(rvec_1 - rvec_2)         # local extrapolation here
                vel_vec = vvec_1 + (1/15)*(vvec_1 - vvec_2)         # local extrapolation here

                pos_list.append(pos_vec) 
                vel_list.append(vel_vec)
                t_list.append(t)

                if rho**(0.25) > 2:     # error trap to avoid making h too large when rho is large
                    h_step = 2*h_step
                else:
                    h_step = h_step*(rho**(0.25))
            else:
                h_step = h_step*(rho**(0.25))

    elif 'r_final' in kwargs:

        r_final = kwargs['r_final']

        while (np.linalg.norm(pos_vec) > r_final):

            r_mid, v_mid = _rk4_step_numba(h_step, pos_vec, vel_vec, A, B)
            rvec_1, vvec_1 = _rk4_step_numba(h_step, r_mid, v_mid, A, B)
            rvec_2, vvec_2 = _rk4_step_numba(2*h_step, pos_vec, vel_vec, A, B)

            error = np.linalg.norm(rvec_2 - rvec_1) + min_error
            rho = (30 * h_step * delta)/error

            if rho >= 1:
                t += 2*h_step                                       # accept current h_step
                pos_vec = rvec_1 + (1/15)*(rvec_1 - rvec_2)         # local extrapolation here
                vel_vec = vvec_1 + (1/15)*(vvec_1 - vvec_2)         # local extrapolation here

                pos_list.append(pos_vec) 
                vel_list.append(vel_vec)
                t_list.append(t)

                if rho**(0.25) > 2:
                    h_step = 2*h_step
                else:
                    h_step = h_step*(rho**(0.25))
            else:
                h_step = h_step*(rho**(0.25))
    else:
        raise ValueError("You must specify either a value for the final radius or final time")
        
    return pos_list,vel_list,t_list

def run_single_simulation(args):

    A, B, h, r_init, v_init, delta, r_final = args

    pos, vel, time_list = rk4_integrate(A, B, h, r_init, v_init, delta, r_final=r_final)
    return time_list[-1]