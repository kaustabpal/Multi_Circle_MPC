from mpc import Agent
import utils
from utils import get_dist
import numpy as np
import matplotlib.pyplot as plt
  
def main():
    p_horizon = 50
    u_horizon = 5
    ### initialize vg and wg
    vg = 0*np.ones((p_horizon,1))
    wg = 0*np.ones((p_horizon,1))

    agent1 = Agent(1,[10,10,np.deg2rad(30)],[50,30,np.deg2rad(30)], vg, wg, p_horizon, u_horizon)
    agent2 = Agent(2,[50,30,np.deg2rad(210)],[10,10,np.deg2rad(210)], vg, wg, p_horizon, u_horizon)

    agent1.obstacles = [agent2]
    agent2.obstacles = [agent1]
    agent1.avoid_obs = True
    agent2.avoid_obs = True

    th = 1.5
    timeout = 200

    dist2 = [] # dist between 1 and 2 
    dist2.append(get_dist(agent1.c_state1, agent2.c_state1))

    rec_video = True
    if(rec_video):
        plt_sv_dir = "tmp/"
        p = 0

    plt.ion()
    plt.show()
    while( ( (np.linalg.norm(agent1.c_state1-agent1.g_state)>th) or \
            (np.linalg.norm(agent2.c_state1-agent2.g_state)>th) ) and timeout>0):
        
        agent1.pred_controls()
        agent2.pred_controls()
        
        for i in range(u_horizon):
            if(np.linalg.norm(agent1.c_state1-agent1.g_state)>th):
                agent1.v = agent1.vg[i]
                agent1.w = agent1.wg[i]
                agent1.v_list.append(agent1.v)
                agent1.x_traj = []
                agent1.y_traj = []
                agent1.get_traj(i)
                agent1.non_hol_update()
            if(np.linalg.norm(agent2.c_state1-agent2.g_state)>th):
                agent2.v = agent2.vg[i]
                agent2.w = agent2.wg[i]
                agent2.v_list.append(agent2.v)
                agent2.x_traj = []
                agent2.y_traj = []
                agent2.get_traj(i)
                agent2.non_hol_update()
            
            dist2.append(get_dist(agent1.c_state1, agent2.c_state1))

            utils.draw([agent1, agent2])
          
            plt.xlim([0,60])
            plt.ylim([0,60])
            plt.title("Agent 1 has Obstacle avoidance")  
            
            if(rec_video):
                plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
                p = p+1
                plt.clf()
            else:
                plt.pause(1e-10)
                plt.clf()

            timeout = timeout - agent1.dt
            
        agent1.vl = agent1.v
        agent1.wl = agent1.w
        agent2.vl = agent2.v
        agent2.wl = agent2.w

    ### Degugging data ###
    agent1.avg_time = sum(agent1.time_list[1:])/len(agent1.time_list[1:])
    agent1.max_time = max(agent1.time_list[1:])
    agent1.min_time = min(agent1.time_list[1:])

    print("Agent-1 avg time: {} secs".format(agent1.avg_time))
    print("Agent-1 max time: {} secs".format(agent1.max_time))
    print("Agent-1 min time: {} secs".format(agent1.min_time))
    print("Minimum distance between the agent1 and agent2:",min(np.array(dist2)))
    if(timeout <= 0):
        print("Stopped because of timeout.")
    ######################

if __name__ == "__main__":
    main()