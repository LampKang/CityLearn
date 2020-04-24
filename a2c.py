from citylearn import  CityLearn
from pathlib import Path
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow as tf
import torch.optim as optim
import random
import torch

tf.keras.backend.set_floatx('float64')

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


gamma = 0.99
update_interval = 5
actor_lr = 0.0005
critic_lr = 0.001



class Buffer:
    def __init__(self):
        self.buffer = []
        
    def append_sample(self, sample):
        self.buffer.append(sample)
        
    def sample(self, sample_size):
        s, a, r, s_next, done = [],[],[],[],[]
        
        if sample_size > len(self.buffer):
            sample_size = len(self.buffer)
            
        rand_sample = random.sample(self.buffer, sample_size)
        for values in rand_sample:
            s.append(values[0])
            a.append(values[1])
            r.append(values[2])
            s_next.append(values[3])
            done.append([4])
        return s, a, r, s_next, done    

    def __len__(self):
         return len(self.buffer)

class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(32, activation='relu')(state_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class RL_Agents_A2C:
    def __init__(self, building_info, observation_spaces = None, action_spaces = None):
        
        #Hyper-parameters
        self.discount = 0.992 #Discount factor
        self.batch_size = 100 #Size of each MINI-BATCH
        self.iterations = 1 # Number of updates of the actor-critic networks every time-step
        self.policy_freq = 2 # Number of iterations after which the actor and target networks are updated
        self.tau = 5e-3 #Rate at which the target networks are updated
        self.lr_init = 1e-3 #5e-2
        self.lr_final = 1e-3 #3e-3
        self.lr_decay_rate = 1/(78*8760)
        self.expl_noise_init = 0.75 # Exploration noise at time-step 0
        self.expl_noise_final = 0.01 # Magnitude of the minimum exploration noise
        self.expl_noise_decay_rate = 1/(290*8760)  # Decay rate of the exploration noise in 1/h
        self.policy_noise = 0.025*0
        self.noise_clip = 0.04*0
        self.max_action = 0.25
        self.min_samples_training = 400 #Min number of tuples that are stored in the batch before the training process begins
        
        # Parameters
        self.device = "cuda:0"
        self.time_step = 0
        self.building_info = building_info # Can be used to create different RL agents based on basic building attributes or climate zones
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.n_buildings = len(observation_spaces)
        self.buffer = {i: Buffer() for i in range(self.n_buildings)}
        self.networks_initialized = False
        
        # Monitoring variables (one per agent)
        self.actor_loss_list = {i: [] for i in range(self.n_buildings)}
        self.critic_loss_list = {i: [] for i in range(self.n_buildings)}
        self.q_val_list = {i: [] for i in range(self.n_buildings)}
        self.q1_list = {i: [] for i in range(self.n_buildings)}
        self.q2_list = {i: [] for i in range(self.n_buildings)}
        self.a_track1 = []
        self.a_track2 = []
        
        #Networks and optimizers (one per agent)
        self.actor, self.critic = {}, {}
        self.state_dim=[]
        self.action_dim=[]
        self.action_bound=[]
        self.std_bound=[]
        for i, (o, a) in enumerate(zip(observation_spaces, action_spaces)):
            # A2C
            self.state_dim.append(o.shape[0])
            self.action_dim.append(a.shape[0])
            self.action_bound.append(a.high[0])
            self.std_bound.append([1e-2, 1.0])
            
            self.actor[i] = Actor(self.state_dim[i], self.action_dim[i], self.action_bound[i], self.std_bound[i])
            self.critic[i] = Critic(self.state_dim[i])

        self.state_batch = []
        self.action_batch = []
        self.td_target_batch = []
        self.advatnage_batch = []
        for i in range(self.n_buildings):
            self.state_batch.append([])
            self.action_batch.append([])
            self.td_target_batch.append([])
            self.advatnage_batch.append([])

    def td_target(self, reward, next_state, done, i):
        if done:
            return reward
        v_value = self.critic[i].model.predict(np.reshape(next_state, [1, self.state_dim[i]]))
        return np.reshape(reward + gamma * v_value[0], [1, 1])

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def select_action(self, states):
   
        actions = []
        for i, state in enumerate(states):
            a = self.actor[i]
            action = a.get_action(state)
            a = np.clip(action, -self.action_bound[i], self.action_bound[i])
            actions.append(a)
        return actions
    
    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        #dones = [dones for _ in range(self.n_buildings)]

        #for i, (s, a, r, s_next, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            #self.buffer[i].append_sample((s,a,r,s_next,done))
            
        for i in range(self.n_buildings):
            #print(i)
           #state, action, reward, next_state, dones_mask = self.buffer[i].sample(self.batch_size)
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            state = np.reshape(state, [1, self.state_dim[i]])
            action = np.reshape(action, [1, self.action_dim[i]])
            next_state = np.reshape(next_state, [1, self.state_dim[i]])
            reward = np.reshape(reward, [1, 1])

            td_target = self.td_target((reward+8)/8, next_state, done, i)
            advantage = self.advatnage(td_target, self.critic[i].model.predict(state))

            self.state_batch[i].append(state)
            self.action_batch[i].append(action)
            self.td_target_batch[i].append(td_target)
            self.advatnage_batch[i].append(advantage)

            if len(self.state_batch[i]) >= update_interval or done:
                states_ = self.list_to_batch(self.state_batch[i])
                actions_ = self.list_to_batch(self.action_batch[i])
                td_targets = self.list_to_batch(self.td_target_batch[i])
                advantages = self.list_to_batch(self.advatnage_batch[i])

                # Compute loss
                actor_loss = self.actor[i].train(states_, actions_, advantages)
                self.actor_loss_list[i].append(actor_loss)
                critic_loss = self.critic[i].train(states_, td_targets)
                self.critic_loss_list[i].append(critic_loss)

                state_batch = []
                action_batch = []
                td_target_batch = []
                advatnage_batch = []
                for i in range(self.n_buildings):
                    self.state_batch.append([])
                    self.action_batch.append([])
                    self.td_target_batch.append([])
                    self.advatnage_batch.append([])



# RL CONTROLLER
#Instantiating the control agent(s)
agents = RL_Agents_A2C(building_info, observations_spaces, actions_spaces)

# Select many episodes for training. In the final run we will set this value to 1 (the buildings run for one year)
episodes = 10

k, c = 0, 0
cost, cum_reward = {}, {}

# The number of episodes can be replaces by a stopping criterion (i.e. convergence of the average reward)
for e in range(episodes):     
    cum_reward[e] = 0
    rewards = []
    state = env.reset()
    done = False
    while not done:

        episode_reward, done = 0, False

        if k%(1000)==0:
            print('hour: '+str(k)+' of '+str(8760*episodes))
            
        action = agents.select_action(state)
        next_state, reward, done, _ = env.step(action)
        #
        agents.add_to_buffer(state, action, reward, next_state, done)
       
        state = next_state
        
        cum_reward[e] += reward[0]
        rewards.append(reward)
        k+=1
        
    cost[e] = env.cost()
    print(cost[e])
    
