from citylearn import  CityLearn
from pathlib import Path
from ppo import Agent
import numpy as np
import time


# Select the climate zone and load environment
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = 'buildings_state_action_space.json'
building_ids = ["Building_1","Building_2","Building_3","Building_4","Building_5","Building_6","Building_7","Building_8","Building_9"]
objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption']

env = CityLearn(data_path, building_attributes, weather_file, solar_profile, building_ids, buildings_states_actions = building_state_actions, cost_function = objective_function)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

# RL CONTROLLER
#Instantiating the control agent(s)
agents = Agent(env, building_info, observations_spaces, actions_spaces)

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
episodes = 10

k, c = 0, 0
cost, cum_reward = {}, {}
start = time.time()
# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
for e in range(episodes):     
    cum_reward[e] = 0
    rewards = []
    states = env.reset()
    done = False

    state_batch = { i : [] for i in range(agents.n_buildings)}
    action_batch = { i : [] for i in range(agents.n_buildings)}
    reward_batch = { i : [] for i in range(agents.n_buildings)}
    old_policy_batch = { i : [] for i in range(agents.n_buildings)}
    while not done:
        if k%(1000)==0:
            print('hour: '+str(k)+' of '+str(8760*episodes))

        actions = []
        log_old_policys = []

        for i in range(agents.n_buildings):
            log_old_policy, action = agents.actor[i].get_action(states[i])
            actions.append(action)
            log_old_policys.append(log_old_policy)
        
        next_states, rewards, done, _ = env.step(actions)
        
        for i in range(agents.n_buildings):
            state = np.reshape(states[i], [1, agents.state_dim[i]])
            action = np.reshape(actions[i], [1,agents.action_dim[i]])
            next_state = np.reshape(next_states[i], [1, agents.state_dim[i]])
            reward = np.reshape(rewards[i], [1, 1])
            log_old_policy = np.reshape(log_old_policys[i], [1, 1])

            state_batch[i].append(state)
            action_batch[i].append(action)
            reward_batch[i].append(reward)
            old_policy_batch[i].append(log_old_policy)
        
        if len(state_batch[0]) >= 5 or done:
            for i in range(agents.n_buildings):
                s = agents.list_to_batch(state_batch[i])
                a = agents.list_to_batch(action_batch[i])
                r = agents.list_to_batch(reward_batch[i])
                old_p = agents.list_to_batch(old_policy_batch[i])
                n_s = np.reshape(next_states[i], [1, agents.state_dim[i]])

                v_values = agents.critic[i].model.predict(s)
                next_v_value = agents.critic[i].model.predict(n_s)
                
                gaes, td_targets = agents.gae_target(r, v_values, next_v_value, done)
                for epoch in range(3):
                    actor_loss = agents.actor[i].train(old_p, s, a, gaes)
                    critic_loss = agents.critic[i].train(s, td_targets)

            state_batch = { i : [] for i in range(agents.n_buildings)}
            action_batch = { i : [] for i in range(agents.n_buildings)}
            reward_batch = { i : [] for i in range(agents.n_buildings)}
            old_policy_batch = { i : [] for i in range(agents.n_buildings)}

        #agents.add_to_buffer(state, action, reward, next_state, done)
        state = next_state
        
        #cum_reward[e] += reward[0]
        #rewards.append(rewards)
        k+=1

    cost[e] = env.cost()
    if c%1==0:
        print(cost[e])
    c+=1

print(time.time() - start)