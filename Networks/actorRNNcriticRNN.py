import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import types


def create_counter_variable(name):
    counter = types.SimpleNamespace()
    counter.var = tf.Variable(0, name=name, trainable=False)
    counter.inc_op = tf.assign(counter.var, counter.var + 1)
    return counter


def create_mean_metrics_from_dict(metrics):
    # Set up summaries for each metric
    update_metrics_ops = []
    summaries = []
    for name, (value, update_op) in metrics.items():
        summaries.append(tf.compat.v1.summary.scalar(name, value))
        update_metrics_ops.append(update_op)
    return tf.summary.merge(summaries), tf.group(update_metrics_ops)

'''This code is taken from https://github.com/bitsauce/Carla-ppo, the recurent neural network was implemented with significant help from another '''
class PolicyGraph():
    """
        Manages the policy computation graph
    """

    def __init__(self, input_states, taken_actions, action_space, scope_name,
                 initial_std=0.4, initial_mean_factor=0.1,
                 pi_hidden_sizes=(500, 300), vf_hidden_sizes=(500, 300), memory=None, critic_memory=None):
        """
            input_states [batch_size, width, height, depth]:
                Input images to predict actions for
            taken_actions [batch_size, num_actions]:
                Placeholder of taken actions for training
            action_space (gym.spaces.Box):
                Continous action space of our agent
            scope_name (string):
            initial_std (float):
                Initial value of the std used in the gaussian policy
                Variable scope name for the policy graph
            initial_mean_factor (float):
                Variance scaling factor for the action mean prediction layer
            pi_hidden_sizes (list):
                List of layer sizes used to construct action predicting MLP
            vf_hidden_sizes (list):
                List of layer sizes used to construct value predicting MLP
        """

        num_actions, action_min, action_max = action_space.shape[0], action_space.low, action_space.high

        with tf.variable_scope(scope_name):
            # Policy branch π(a_t | s_t; θ)
            #next 4 lines is actor only recurrent
            if memory is None:
                self.pi = self.build_mlp(input_states, hidden_sizes=pi_hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)
            else:
                self.pi, _ = self.build_rnn(input_states, memory, hidden_sizes=pi_hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)

            self.action_mean = tf.layers.dense(self.pi, num_actions,
                                               activation=tf.nn.tanh,
                                               kernel_initializer=tf.initializers.variance_scaling(scale=initial_mean_factor),
                                               name="action_mean")

            self.action_mean = action_min + ((self.action_mean + 1) / 2) * (action_max - action_min)
            self.action_logstd = tf.Variable(np.full((num_actions), np.log(initial_std), dtype=np.float32), name="action_logstd")

            #next 4 lines are critic
            # Value branch V(s_t; θ)
            if vf_hidden_sizes is None:
                self.vf = self.pi # Share features if None
            else:
                if critic_memory is None:
                    self.vf = self.build_mlp(input_states, hidden_sizes=vf_hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)
                else:
                    self.vf, _ = self.build_rnn(input_states, critic_memory, hidden_sizes=vf_hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.relu)
            #critic componenet, since output is one
            self.value = tf.squeeze(tf.layers.dense(self.vf, 1, activation=None, name="value"), axis=-1)

            # Create graph for sampling actions
            self.action_normal  = tfp.distributions.Normal(self.action_mean, tf.exp(self.action_logstd), validate_args=True)
            self.sampled_action = tf.squeeze(self.action_normal.sample(1), axis=0)

            # Clip action space to min max
            self.sampled_action = tf.clip_by_value(self.sampled_action, action_min, action_max)

            # Get the log probability of taken actions
            # log π(a_t | s_t; θ)
            self.action_log_prob = tf.reduce_sum(self.action_normal.log_prob(taken_actions), axis=-1, keepdims=True)

    def build_mlp(self, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return x
#h is states
    def build_rnn(self, x, h, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for i in hidden_sizes[:-1]:
            x, h = tf.keras.layers.SimpleRNNCell(units=i)(x, [h,])
        return x, h

    def build_lstm(self, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        #rah = hidden_sizes[:-1][:1][0]
        for h in hidden_sizes[:-1]:
            x = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.layers.dense(x, units=500, activation=activation)
        return x


class PPO():
    """
        Proximal policy gradient model class
    """

    def __init__(self, input_shape, action_space,
                 learning_rate=3e-4, lr_decay=0.998, epsilon=0.2,
                 value_scale=0.5, entropy_scale=0.01, initial_std=0.4,
                 model_dir="./"):
        """
            input_shape [3]:
                Shape of input images as a tuple (width, height, depth)
            action_space (gym.spaces.Box):
                Continous action space of our agent
            learning_rate (float):
                Initial learning rate
            lr_decay (float):
                Learning rate decay exponent
            epsilon (float):
                PPO clipping parameter
            value_scale (float):
                Value loss scale factor
            entropy_scale (float):
                Entropy loss scale factor
            initial_std (float):
                Initial value of the std used in the gaussian policy
            model_dir (string):
                Directory to output the trained model and log files
        """
        pi_hidden_sizes = (500, 300)
        vf_hidden_sizes=(500, 300)
        num_actions = action_space.shape[0]

        # Create counters
        self.train_step_counter   = create_counter_variable(name="train_step_counter")
        self.predict_step_counter = create_counter_variable(name="predict_step_counter")
        self.episode_counter      = create_counter_variable(name="episode_counter")

        # Create placeholders
        self.input_states  = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
        #component h for rnn
        self.memory = tf.placeholder(shape=(None, pi_hidden_sizes[0]), dtype=tf.float32, name="input_memory_placeholder")
        self.critic_memory = tf.placeholder(shape=(None, vf_hidden_sizes[0]), dtype=tf.float32, name="input_critic_memory_placeholder")
        self.taken_actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")
        self.returns   = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")

        # Create policy graphs
        self.policy        = PolicyGraph(self.input_states, self.taken_actions, action_space, "policy", initial_std=initial_std, pi_hidden_sizes=pi_hidden_sizes, memory=self.memory, vf_hidden_sizes=vf_hidden_sizes, critic_memory=self.critic_memory)
        self.policy_old    = PolicyGraph(self.input_states, self.taken_actions, action_space, "policy_old", initial_std=initial_std, pi_hidden_sizes=pi_hidden_sizes, memory=self.memory, vf_hidden_sizes=vf_hidden_sizes, critic_memory=self.critic_memory)

        # Calculate ratio:
        # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
        # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
        # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
        self.prob_ratio = tf.exp(self.policy.action_log_prob - self.policy_old.action_log_prob)

        # Policy loss
        adv = tf.expand_dims(self.advantage, axis=-1)
        self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv, tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv))

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.math.squared_difference(self.policy.value, self.returns)) * value_scale

        # Entropy loss
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1)) * entropy_scale

        # Total loss
        self.loss = -self.policy_loss + self.value_loss - self.entropy_loss

        # Policy parameters
        policy_params     = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
        policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
        assert(len(policy_params) == len(policy_old_params))
        for src, dst in zip(policy_params, policy_old_params):
            assert(src.shape == dst.shape)

        # Minimize loss
        self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, self.episode_counter.var, 1, lr_decay, staircase=True)
        self.optimizer     = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step    = self.optimizer.minimize(self.loss, var_list=policy_params)

        # Update network parameters
        self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

        # Set up episodic metrics
        metrics = {}
        metrics["train_loss/policy"] = tf.compat.v1.metrics.mean(self.policy_loss)
        metrics["train_loss/value"] = tf.compat.v1.metrics.mean(self.value_loss)
        metrics["train_loss/entropy"] = tf.compat.v1.metrics.mean(self.entropy_loss)
        metrics["train_loss/loss"] = tf.compat.v1.metrics.mean(self.loss)
        for i in range(num_actions):
            metrics["train_actor/action_{}/taken_actions".format(i)] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.taken_actions[:, i]))
            metrics["train_actor/action_{}/mean".format(i)] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.policy.action_mean[:, i]))
            metrics["train_actor/action_{}/std".format(i)] = tf.compat.v1.metrics.mean(tf.reduce_mean(tf.exp(self.policy.action_logstd[i])))
        metrics["train/prob_ratio"] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.prob_ratio))
        metrics["train/returns"] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.returns))
        metrics["train/advantage"] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.advantage))
        metrics["train/learning_rate"] = tf.compat.v1.metrics.mean(tf.reduce_mean(self.learning_rate))
        self.episodic_summaries, self.update_metrics_op = create_mean_metrics_from_dict(metrics)

        # Set up stepwise training summaries
        summaries = []
        for i in range(num_actions):
            summaries.append(tf.compat.v1.summary.histogram("train_actor_step/action_{}/taken_actions".format(i), self.taken_actions[:, i]))
            summaries.append(tf.compat.v1.summary.histogram("train_actor_step/action_{}/mean".format(i), self.policy.action_mean[:, i]))
            summaries.append(tf.compat.v1.summary.histogram("train_actor_step/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
        summaries.append(tf.compat.v1.summary.histogram("train_step/input_states", self.input_states))
        summaries.append(tf.compat.v1.summary.histogram("train_step/prob_ratio", self.prob_ratio))
        self.stepwise_summaries = tf.compat.v1.summary.merge(summaries)

        # Set up stepwise prediction summaries
        summaries = []
        for i in range(num_actions):
            summaries.append(tf.compat.v1.summary.scalar("predict_actor/action_{}/sampled_action".format(i), self.policy.sampled_action[0, i]))
            summaries.append(tf.compat.v1.summary.scalar("predict_actor/action_{}/mean".format(i), self.policy.action_mean[0, i]))
            summaries.append(tf.compat.v1.summary.scalar("predict_actor/action_{}/std".format(i), tf.exp(self.policy.action_logstd[i])))
        self.stepwise_prediction_summaries = tf.compat.v1.summary.merge(summaries)

            # Setup model saver and dirs
        self.saver = tf.train.Saver()
        self.model_dir = model_dir
        self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
        self.log_dir        = "{}/logs/".format(self.model_dir)
        self.video_dir      = "{}/videos/".format(self.model_dir)
        self.dirs = [self.checkpoint_dir, self.log_dir, self.video_dir]
        for d in self.dirs: os.makedirs(d, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run([tf.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        else:
            self.sess = sess

        if init_logging:
            self.train_writer = tf.compat.v1.summary.FileWriter(self.log_dir, self.sess.graph)

    def save(self):
        model_checkpoint = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.episode_counter.var)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir, latest_filename=None)
        print(model_checkpoint,"AND!", self.checkpoint_dir, self.sess.__str__())
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False

    def train(self, input_states, taken_actions, returns, advantage):
        batch_size = input_states.shape[0]
        memory = np.zeros((batch_size, self.memory.shape[1]), dtype=np.float32)
        critic_memory = np.zeros((batch_size, self.critic_memory.shape[1]), dtype=np.float32)
        _, _, summaries, step_idx = \
            self.sess.run([self.train_step, self.update_metrics_op, self.stepwise_summaries, self.train_step_counter.var],
                feed_dict={
                    self.input_states: input_states,
                    self.memory: memory,
                    self.critic_memory: critic_memory,
                    self.taken_actions: taken_actions,
                    self.returns: returns,
                    self.advantage: advantage
                }
            )
        self.train_writer.add_summary(summaries, step_idx)
        self.sess.run(self.train_step_counter.inc_op) # Inc step counter

    def predict(self, input_states, greedy=False, write_to_summary=False):

        # Extend input axis 0 if no batch dim
        input_states = np.asarray(input_states)
        if len(input_states.shape) != 2:
            input_states = [input_states]
            memory = np.zeros((1, self.memory.shape[1]), dtype=np.float32)
            critic_memory = np.zeros((1, self.critic_memory.shape[1]), dtype=np.float32)
        else:
            batch_size = input_states.shape[0]
            memory = np.zeros((batch_size, self.memory.shape[1]), dtype=np.float32)
            critic_memory = np.zeros((batch_size, self.critic_memory.shape[1]), dtype=np.float32)

        # Predict action
        action = self.policy.action_mean if greedy else self.policy.sampled_action
        sampled_action, value, summaries, step_idx = \
            self.sess.run([action, self.policy.value, self.stepwise_prediction_summaries, self.predict_step_counter.var],
                feed_dict={self.input_states: input_states, self.memory: memory, self.critic_memory: critic_memory}
            )

        if write_to_summary:
            self.train_writer.add_summary(summaries, step_idx)
            self.sess.run(self.predict_step_counter.inc_op)

        # Squeeze output if output has one element
        if len(input_states) == 1:
            return sampled_action[0], value[0]
        return sampled_action, value

    def get_episode_idx(self):
        return self.sess.run(self.episode_counter.var)

    def get_train_step_idx(self):
        return self.sess.run(self.train_step_counter.var)

    def get_predict_step_idx(self):
        return self.sess.run(self.predict_step_counter.var)

    def write_value_to_summary(self, summary_name, value, step):
        summary = tf.Summary()
        summary.value.add(tag=summary_name, simple_value=value)
        self.train_writer.add_summary(summary, step)

    def write_dict_to_summary(self, summary_name, params, step):
        summary_op = tf.compat.v1.summary.text(summary_name, tf.stack([tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]))
        self.train_writer.add_summary(self.sess.run(summary_op))

    def write_episodic_summaries(self):
        self.train_writer.add_summary(self.sess.run(self.episodic_summaries), self.get_episode_idx())
        self.sess.run([self.episode_counter.inc_op, tf.local_variables_initializer()])

    def update_old_policy(self):
        self.sess.run(self.update_op)
