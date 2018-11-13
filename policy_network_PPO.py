import os.path
import numpy as np
import tensorflow as tf

OBSERVATIONS_SIZE = 6400


class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32,
                                           [None, OBSERVATIONS_SIZE])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        self.OLD_ACTION = tf.placeholder(tf.float32, [None,1])   # old negative log action probability

        h = tf.layers.dense(
            self.observations,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Train based on the log probability of the sampled action.
        # 
        # The idea is to encourage actions taken in rounds where the agent won,
        # and discourage actions in rounds where the agent lost.
        # More specifically, we want to increase the log probability of winning
        # actions, and decrease the log probability of losing actions.
        #
        # Which direction to push the log probability in is controlled by
        # 'advantage', which is the reward for each action in each round.
        # Positive reward pushes the log probability of chosen action up;
        # negative reward pushes the log probability of the chosen action down.
        #self.loss = tf.losses.log_loss(
        #    labels=self.sampled_actions,
        #    predictions=self.up_probability,
        #    weights=self.advantage)

        #epsilon = 1e-7 
        #self.loss =  -1*tf.reduce_mean( (self.sampled_actions * tf.log( self.up_probability + epsilon ) + (1 - self.sampled_actions) * tf.log( 1- self.up_probability + epsilon ) ) * self.advantage ) 

        # policy gradient loss (clipped surrogate objective)

        epsilon = 1e-7 
        OLD_NEG_LOGP_ACTION = -1*( self.sampled_actions * tf.log( self.OLD_ACTION + epsilon ) + (1 - self.sampled_actions) * tf.log( 1- self.OLD_ACTION + epsilon ) )
        CLIP_RANGE = 0.2

        neglogpac = -1*( self.sampled_actions * tf.log( self.up_probability + epsilon ) + (1 - self.sampled_actions) * tf.log( 1- self.up_probability + epsilon ) )
        
        ratio = tf.exp(OLD_NEG_LOGP_ACTION - neglogpac) # equivalent to: ratio = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
        losses1 =  self.advantage * ratio
        losses2 =  self.advantage * tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
        self.loss = tf.reduce_mean(-tf.minimum(losses1, losses2))



        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards,old_network_actions = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        old_network_actions = np.vstack(old_network_actions)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards,
            self.OLD_ACTION : old_network_actions
        }
        print(rewards)
        loss , _ = self.sess.run([self.loss, self.train_op], feed_dict)
        print("The loss is : ")
        print(loss)
        
