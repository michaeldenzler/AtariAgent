import tensorflow as tf
import numpy as np
import gym
from collections import deque
from atari_wrappers import wrap_atari_deepmind
import matplotlib.pyplot as plt

# start with epsilon = 1, over the first 1mio steps linearly decrease to 0.1, then stay at 0.1
def update_epsilon(e):
    return max(e - 0.9/1000000, 0.1)

tf.reset_default_graph()
tf.set_random_seed(seed=0)

# wrap the environment to a 84x84x4 image
env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', True)

num_actions = 4 # action space = [0, 1, 2, 3]
learning_rate = 0.0001
decay = 0.99

X = tf.placeholder(tf.float32, [None, 84, 84, 4])
Y = tf.placeholder(tf.float32, [None])
A = tf.placeholder(tf.int64, [None, 2])

init_weights = tf.variance_scaling_initializer()
init_biases = tf.zeros_initializer()

## ONLINE Q-NETWORK
# Convolutional layer 1
Wo_conv1 = tf.Variable(init_weights([8, 8, 4, 32]))  # 32 filters a (8x8) with 4 channels each
bo_conv1 = tf.Variable(init_biases((32,)))
Ao_conv1 = tf.nn.relu(tf.nn.conv2d(X, Wo_conv1, strides=[1, 4, 4, 1], padding='SAME') + bo_conv1)  # ? * 21 * 21 * 32

# Convolutional layer 2
Wo_conv2 = tf.Variable(init_weights([4, 4, 32, 64]))  # 64 filters a (4x4) with 32 channels each
bo_conv2 = tf.Variable(init_biases((64,)))
Ao_conv2 = tf.nn.relu(tf.nn.conv2d(Ao_conv1, Wo_conv2, strides=[1, 2, 2, 1], padding='SAME') + bo_conv2)  # ? * 11 * 11 * 64

# Convolutional layer 3
Wo_conv3 = tf.Variable(init_weights([3, 3, 64, 64]))  # 64 filters a (3x3) with 64 channels each
bo_conv3 = tf.Variable(init_biases((64,)))
Ao_conv3 = tf.nn.relu(tf.nn.conv2d(Ao_conv2, Wo_conv3, strides=[1, 1, 1, 1], padding='SAME') + bo_conv3)  # ? * 11 * 11 * 64

Ao_conv3_flat = tf.reshape(Ao_conv3, [-1, 11 * 11 * 64]) # ? * 7744

# Fully Connected layer 1
Wo_fc1 = tf.Variable(init_weights([11 * 11 * 64, 512]))
bo_fc1 = tf.Variable(init_biases((512,)))
Ao_fc1 = tf.nn.relu(tf.matmul(Ao_conv3_flat, Wo_fc1) + bo_fc1) # ? * 512

# Fully connected layer 2
Wo_fc2 = tf.Variable(init_weights([512, num_actions]))
bo_fc2 = tf.Variable(init_biases((num_actions,)))
Zo = tf.matmul(Ao_fc1, Wo_fc2) + bo_fc2  # ? * num_actions


## TARGET Q-NETWORK
# Convolutional layer 1
Wt_conv1 = tf.Variable(init_weights([8, 8, 4, 32]))  # 32 filters a (8x8) with 4 channels each
bt_conv1 = tf.Variable(init_biases((32,)))
At_conv1 = tf.nn.relu(tf.nn.conv2d(X, Wt_conv1, strides=[1, 4, 4, 1], padding='SAME') + bt_conv1)  # ? * 21 * 21 * 32

# Convolutional layer 2
Wt_conv2 = tf.Variable(init_weights([4, 4, 32, 64]))  # 64 filters a (4x4) with 32 channels each
bt_conv2 = tf.Variable(init_biases((64,)))
At_conv2 = tf.nn.relu(
    tf.nn.conv2d(At_conv1, Wt_conv2, strides=[1, 2, 2, 1], padding='SAME') + bt_conv2)  # ? * 11 * 11 * 64

# Convolutional layer 3
Wt_conv3 = tf.Variable(init_weights([3, 3, 64, 64]))  # 64 filters a (3x3) with 64 channels each
bt_conv3 = tf.Variable(init_biases((64,)))
At_conv3 = tf.nn.relu(
    tf.nn.conv2d(At_conv2, Wt_conv3, strides=[1, 1, 1, 1], padding='SAME') + bt_conv3)  # ? * 11 * 11 * 64

At_conv3_flat = tf.reshape(At_conv3, [-1, 11 * 11 * 64])  # ? * 7744

# Fully Connected layer 1
Wt_fc1 = tf.Variable(init_weights([11 * 11 * 64, 512]))
bt_fc1 = tf.Variable(init_biases((512,)))
At_fc1 = tf.nn.relu(tf.matmul(At_conv3_flat, Wt_fc1) + bt_fc1)  # ? * 512

# Fully connected layer 2
Wt_fc2 = tf.Variable(init_weights([512, num_actions]))
bt_fc2 = tf.Variable(init_biases((num_actions,)))
Zt = tf.matmul(At_fc1, Wt_fc2) + bt_fc2  # ? * num_actions

# Squared error
loss = tf.reduce_sum(tf.square(Y - tf.gather_nd(Zo, A)))

optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
train = optimizer.minimize(loss)

# define hyperparameters
M = 10000
gamma = 0.99
N = 2000000
init_epsilon = 1
n = 4
B = 32
C = 10000
E_eval = 100000
epsilon_eval = 0.001

# update target network weights given online network weights
update_1 = tf.assign(Wt_conv1, Wo_conv1)
update_2 = tf.assign(bt_conv1, bo_conv1)
update_3 = tf.assign(Wt_conv2, Wo_conv2)
update_4 = tf.assign(bt_conv2, bo_conv2)
update_5 = tf.assign(Wt_conv3, Wo_conv3)
update_6 = tf.assign(bt_conv3, bo_conv3)
update_7 = tf.assign(Wt_fc1, Wo_fc1)
update_8 = tf.assign(bt_fc1, bo_fc1)
update_9 = tf.assign(Wt_fc2, Wo_fc2)
update_10 = tf.assign(bt_fc2, bo_fc2)

# Allows saving model to disc
saver = tf.train.Saver()

# initialize replay buffer as deque
replay_buffer = deque(maxlen=M)

session = tf.Session()
session.run(tf.global_variables_initializer())

# update target network
session.run([update_1, update_2, update_3, update_4, update_5,
              update_6, update_7, update_8, update_9, update_10])
e = init_epsilon
i = 0
episode = 0
reward_episodes = []
losses = []
avg_scores = []
while i < N:
    # initialize new episode
    state = env.reset()
    reward_episode = 0
    while True:
        # convert state from Lazy frame into numpy array
        state_array = np.array(state).reshape([1, 84, 84, 4])

        # e-greedy action selection
        if np.random.random() < 1 - e:
            action = np.argmax(session.run(Zo, {X: state_array}))
        else:
            action = env.action_space.sample()
        e = update_epsilon(e)  # decrease epsilon down to 0.1 after 1 mio steps
        next_state, reward, termination, _ = env.step(action)  # make a step
        next_state_array = np.array(next_state).reshape([1, 84, 84, 4]) # convert next_state from Lazy frame into numpy array
        reward_episode += reward # count rewards per episode so far
        replay_buffer.append([state_array, action, reward, next_state_array, termination])  # add encountered situation to buffer
        state = next_state  # define next_state as current state
        i += 1  # iteration count up
        if i >= M:
            if i % n == 0:  # train model every 4 steps
                indices = np.random.randint(M, size=B)  # draw random indices to pick training values out of buffer
                states = np.zeros([B, 84, 84, 4])
                actions = np.zeros([B, 2])
                r = np.zeros([B,])
                next_states = np.zeros([B, 84, 84, 4])
                d = np.zeros([B,])

                for k in range(B):
                    sample = replay_buffer[indices[k]]  # get sample
                    states[k] = np.array(sample[0]).reshape(1, 84, 84, 4)
                    actions[k, 0] = k
                    actions[k, 1] = sample[1]
                    r[k] = sample[2]
                    next_states[k] = np.array(sample[3]).reshape(1, 84, 84, 4)
                    d[k] = sample[4]
                Q_sa_target = session.run(Zt, {X: next_states})  # compute Q_target(s',a',theta')
                Q_sa_target_max = np.max(Q_sa_target, axis=1)  # pick maximal Q_target(s',a',theta')
                y = r + gamma * Q_sa_target_max * (1 - d)  # define target values y

                l, _ = session.run([loss, train], {X: states, Y: y, A: actions})  # compute loss and train model

                if i % 20000 == 0:
                    print('Iteration: {0}. Loss: {1}.'.format(i, l))  # for loss tracking

            # update target network
            if i % C == 0:
                losses = np.append(losses, l)
                session.run([update_1, update_2, update_3, update_4, update_5,
                              update_6, update_7, update_8, update_9, update_10])

            # evaluation play
            if i % E_eval == 0:
                # initiate environment without clipped rewards
                env_eval = wrap_atari_deepmind('BreakoutNoFrameskip-v4', False)
                state_eval = env_eval.reset()
                t_plays = 0  # count number of plays so far
                total_score = 0  # store total score so far
                while t_plays < 30:
                    t_epochs = 0  # count number of epochs per play
                    reward_play_eval = 0  # store rewards of current play
                    while t_epochs < 5:
                        # e-greedy action selection
                        if np.random.random() < 1 - epsilon_eval:
                            state_eval = np.array(state_eval).reshape([1, 84, 84, 4])
                            action_eval = np.argmax(session.run(Zo, {X: state_eval}))
                        else:
                            action_eval = env_eval.action_space.sample()

                        state_eval, reward_eval, termination_eval, _ = env_eval.step(action_eval)
                        reward_play_eval += reward_eval  # sum up rewards of current play
                        # if terminal state, initiate next epoch
                        if termination_eval:
                            t_epochs += 1
                            state_eval = env_eval.reset()
                    score = reward_play_eval
                    total_score += score  # sum up total scores
                    t_plays += 1
                avg_score = total_score / 30  # compute average score over 30 plays
                avg_scores = np.append(avg_scores, avg_score)

                print('Eval at time step {0}. Avg score over 30 plays: {1}.'.format(i, avg_score))

        # if terminal state, increase episode count and break out of current episode loop
        if termination:
            if episode % 200 == 0:
                print('Episode: {0}. Reward Episode: {1}'.format(episode, reward_episode))
            reward_episodes = np.append(reward_episodes, reward_episode)
            episode += 1
            break

saver.save(session, './drive/My Drive/DeepLearning/assignment4.ckpt')
session.close()

# write all reporting data to file
f = open("./drive/My Drive/DeepLearning/avg_scores.csv","w+")
np.savetxt('./drive/My Drive/DeepLearning/avg_scores.csv', [avg_scores], delimiter=',', fmt='%1.3f')
f.close()

f = open("./drive/My Drive/DeepLearning/losses.csv","w+")
np.savetxt('./drive/My Drive/DeepLearning/losses.csv', [losses], delimiter=',', fmt='%1.3f')
f.close()

f = open("./drive/My Drive/DeepLearning/rewards_episode.csv","w+")
np.savetxt('./drive/My Drive/DeepLearning/rewards_episode.csv', [reward_episodes], delimiter=',', fmt='%1.3f')
f.close()

### Reporting results ###

# plot average scores
avg_scores = np.genfromtxt('./drive/My Drive/DeepLearning/avg_scores.csv', delimiter=',')
plt.plot(avg_scores, label='average score over 30 epochs')
plt.legend()
plt.show()

# plot losses
losses = np.genfromtxt('./drive/My Drive/DeepLearning/losses.csv', delimiter=',')
plt.plot(losses, label='losses')
plt.legend()
plt.show()

# plot moving average of rewards per episode
reward_episodes = np.genfromtxt('./drive/My Drive/DeepLearning/rewards_episode.csv', delimiter=',')
N = 30
reward_moving_avg = np.convolve(reward_episodes, np.ones((N,))/N, mode='valid')
plt.plot(reward_moving_avg, label='moving average for rewards per episode')
plt.legend()
plt.show()

# re-initialize stored session
session = tf.Session()
saver.restore(session, './drive/My Drive/DeepLearning/assignment4.ckpt')
# wrap environment and create video of one episode played
env_video = wrap_atari_deepmind('BreakoutNoFrameskip-v4', False)
env_video = gym.wrappers.Monitor(env_video, './drive/My Drive/DeepLearning/video1', force=True)
state_video = env_video.reset()
while True:
    state_video_array = np.array(state_video).reshape([1, 84, 84, 4])
    action_video = np.argmax(session.run(Zo, {X: state_video_array}))
    state_video, reward_video, termination_video, _ = env_video.step(action_video)
    if termination_video:
        state_video = env_video.reset()
        break
env_video.close()

session.close()