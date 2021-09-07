import numpy as np 
import matplotlib.pyplot as plt

def get_dist(a_state, o_state):
    d = np.sqrt((o_state[0] - a_state[0])**2 + (o_state[1] - a_state[1])**2)
    return d       

def draw_circle(x, y, radius):
    th = np.arange(0,2*np.pi,0.01)
    xunit = radius * np.cos(th) + x
    yunit = radius * np.sin(th) + y
    return xunit, yunit  

def draw(a_list):
    for i in range(len(a_list)):
        a = a_list[i]
        if(a.avoid_obs):
            col = 'g'
        else:
            col = 'r'
        x1, y1 = draw_circle(a.c_state1[0], a.c_state1[1], a.a_radius)
        x2, y2 = draw_circle(a.c_state2[0], a.c_state2[1], a.a_radius)
        x3, y3 = draw_circle(a.c_state3[0], a.c_state3[1], a.a_radius)
        
        plt.plot(x1, y1, col, linewidth=1)
        plt.plot(x2, y2, col, linewidth=1)
        plt.plot(x3, y3, col, linewidth=1)

        plt.annotate(str(a.agent_id), xy=(a.c_state1[0], a.c_state1[1]+2.0))

        plt.scatter(a.g_state[0], a.g_state[1], marker='x', color='r')
        plt.scatter(a.x_traj, a.y_traj, marker='.', color='cyan', s=1)
        plt.plot([a.c_state1[0], a.g_state[0]], [a.c_state1[1],a.g_state[1]], linestyle='dotted', c='k')
