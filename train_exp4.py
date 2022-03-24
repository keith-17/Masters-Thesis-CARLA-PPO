import os
import random
import shutil
import sys
import glob

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re
import scipy.signal
import gym

'''import custom wrappers and scripts'''
from wrappers2 import vector
from reward_functions2 import reward_functions
from actorRNNcriticRNNexp4 import PPO
from run_eval import run_eval
from train_vaeT1_aug import ConvVAE

from environments_exp4 import CarlaRouteEnv

from environments_exp4 import CarlaLapEnv

'''Connect to CARLA client'''
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

#funcitn to compute advantages
def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrap_values])
    terminals = np.array(terminals)
    deltas = rewards + (1.0 - terminals) * gamma * values[1:] - values[:-1]
    return scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]

#load CVAE
def load_vae(model_dir, z_dim=None, model_type=None):
    """
        Loads and returns a pretrained VAE
    """

    # Parse z_dim and model_type from name if None
    if z_dim is None: z_dim = int(re.findall("zdim(\d+)", model_dir)[0])
    VAEClass = ConvVAE

    target_depth = 3

    # Load pre-trained variational autoencoder
    vae_source_shape = np.array([80, 160, 3])
    vae = VAEClass(source_shape=vae_source_shape,
                   target_shape=np.array([80, 160, target_depth]),
                   z_dim=z_dim, models_dir="vae",
                   model_dir=model_dir,
                   training=False)
    vae.init_session(init_logging=False)
    if not vae.load_latest_checkpoint():
        raise Exception("Failed to load VAE")
    return vae

#encode different sttates for observations
def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    # Turn into bool array for performance
    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "brake" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "accel_x" in measurements_to_include,
                     "accel_y" in measurements_to_include,
                     "yaw_vel" in measurements_to_include]#,
                     #"Lidar" in measurements_to_include]

    def encode_state(env):
        # Encode image with VAE
        frame = preprocess_frame(env.observation)
        encoded_state = vae.encode([frame])[0]

        # Append measurements
        measurements = []
        if measure_flags[0]: measurements.append(env.vehicle.control.steer)
        if measure_flags[1]: measurements.append(env.vehicle.control.throttle)
        if measure_flags[2]: measurements.append(env.vehicle.control.brake)
        if measure_flags[3]: measurements.append(env.vehicle.get_speed())
        if measure_flags[4]: measurements.append(env.observation_momentum["IMU"].accelerometer.x)
        if measure_flags[5]: measurements.append(env.observation_momentum["IMU"].accelerometer.y)
        if measure_flags[6]: measurements.append(env.observation_momentum["IMU"].gyroscope.z)

        encoded_state = np.append(encoded_state, measurements)

        return encoded_state
    return encode_state

#normalise frame
def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame

#training routine
def train(args, restart=False):
    # Set seeds
    if isinstance(args.seed, int):
        tf.compat.v1.random.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(0)

    # Load VAE
    vae = load_vae(args.vae_model, args.vae_z_dim, args.vae_model_type)

    # Override params for logging
    params["vae_z_dim"] = vae.z_dim

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "brake", "speed","accel_x","accel_y","yaw_vel"])
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)


    best_eval_reward = -float("inf")

    action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer, throttle/brake

    # Environment constants
    input_shape = np.array([vae.z_dim + len(measurements_to_include)])
    num_actions = action_space.shape[0]
    print(args.model_name)
    # Create model
    print("Creating model")
    model = PPO(input_shape, action_space,
                learning_rate=args.learning_rate, lr_decay=args.lr_decay,
                epsilon=args.ppo_epsilon, initial_std=args.initial_std,
                value_scale=args.value_scale, entropy_scale=args.entropy_scale,
                model_dir=os.path.join("models", args.model_name))

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(args.model_name))
            if answer.upper() == "C":
                pass
            elif answer.upper() == "R":
                restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(args.model_name))

    if restart:
        shutil.rmtree(model.model_dir)
        for d in model.dirs:
            os.makedirs(d)
    model.init_session()
    if not restart:
        model.load_latest_checkpoint()
    model.write_dict_to_summary("hyperparameters", params, 0)
    interlace_episode = 1
    create_environment = 1
    CarlaEnv = CarlaLapEnv
    create_environment = 2
    # Create env
    print("Creating environment")
    env = CarlaEnv(args,
                    reward_fn=reward_functions[args.reward_fn],
                    encode_state_fn=encode_state_fn,
                    synchronous=args.synchronous,
                    fps=args.fps,
                    action_smoothing=args.action_smoothing)
    # For every episode
    while args.num_episodes <= 0 or model.get_episode_idx() < args.num_episodes:
        if interlace_episode % args.switch_environment == 0:
            if create_environment == 1:
                CarlaEnv = CarlaLapEnv
                create_environment = 2
                # Create env
                print("Creating environment")
                env = CarlaEnv(args,
                                reward_fn=reward_functions[args.reward_fn],
                                encode_state_fn=encode_state_fn,
                                synchronous=args.synchronous,
                                fps=args.fps,
                                action_smoothing=args.action_smoothing)
            elif create_environment == 2:
                CarlaEnv = CarlaRouteEnv
                # Create env
                print("Creating environment")
                env = CarlaEnv(args,
                                reward_fn=reward_functions[args.reward_fn],
                                encode_state_fn=encode_state_fn,
                                synchronous=args.synchronous,
                                fps=args.fps,
                                action_smoothing=args.action_smoothing)
                create_environment = 1

        episode_idx = model.get_episode_idx()

        # Run evaluation periodically
        if episode_idx % args.eval_interval == 0:
            video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
            eval_reward = run_eval(env, model, video_filename=video_filename)
            model.write_value_to_summary("eval/reward", eval_reward, episode_idx)
            model.write_value_to_summary("eval/distance_traveled", env.distance_traveled, episode_idx)
            model.write_value_to_summary("eval/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
            model.write_value_to_summary("eval/center_lane_deviation", env.center_lane_deviation, episode_idx)
            model.write_value_to_summary("eval/average_center_lane_deviation", env.center_lane_deviation / env.step_count, episode_idx)
            model.write_value_to_summary("eval/distance_over_deviation", env.distance_traveled / env.center_lane_deviation, episode_idx)
            if eval_reward > best_eval_reward:
                model.save()
                best_eval_reward = eval_reward

        # Reset environment
        state, terminal_state, total_reward = env.reset(), False, 0

        # While episode not done
        print(f"Episode {episode_idx} (Step {model.get_train_step_idx()})")
        while not terminal_state:
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            for _ in range(args.horizon):
                action, value = model.predict(state, write_to_summary=True)

                # Perform action
                new_state, reward, terminal_state, info = env.step(action)

                if info["closed"] == True:
                    exit(0)

                env.extra_info.extend([
                    "Episode {}".format(episode_idx),
                    "Training...",
                    "",
                    "Value:  % 20.2f" % value
                ])

                env.render()
                total_reward += reward

                # Store state, action and reward
                states.append(state)         # [T, *input_shape]
                taken_actions.append(action) # [T,  num_actions]
                values.append(value)         # [T]
                rewards.append(reward)       # [T]
                dones.append(terminal_state) # [T]
                state = new_state

                if terminal_state:
                    break

            # Calculate last value (bootstrap value)
            _, last_values = model.predict(state) # []

            # Compute GAE
            advantages = compute_gae(rewards, values, last_values, dones, args.discount_factor, args.gae_lambda)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten arrays
            states        = np.array(states)
            taken_actions = np.array(taken_actions)
            returns       = np.array(returns)
            advantages    = np.array(advantages)

            T = len(rewards)
            assert states.shape == (T, *input_shape)
            assert taken_actions.shape == (T, num_actions)
            assert returns.shape == (T,)
            assert advantages.shape == (T,)

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(args.num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / args.batch_size))):
                    # Sample mini-batch randomly
                    begin = i * args.batch_size
                    end   = begin + args.batch_size
                    if end > num_samples:
                        end = None
                    mb_idx = indices[begin:end]

                    # Optimize network
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])

        # Write episodic values
        model.write_value_to_summary("train/reward", total_reward, episode_idx)
        model.write_value_to_summary("train/distance_traveled", env.distance_traveled, episode_idx)
        model.write_value_to_summary("train/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
        model.write_value_to_summary("train/center_lane_deviation", env.center_lane_deviation, episode_idx)
        model.write_value_to_summary("train/average_center_lane_deviation", env.center_lane_deviation / env.step_count, episode_idx)
        model.write_value_to_summary("train/distance_over_deviation", env.distance_traveled / env.center_lane_deviation, episode_idx)
        model.write_episodic_summaries()
        interlace_episode+=1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains a CARLA agent with PPO")
    #client and hud parameters
    parser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    parser.add_argument("--viewer_res", default="1280x720", type=str, help="Window resolution (default: 1280x720)")
    parser.add_argument("--obs_res", default="160x80", type=str, help="Output resolution (default: same as --res)")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1, help="Per-episode exponential learning rate decay")
    parser.add_argument("--discount_factor", type=float, default=0.9, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0, help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")
    parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to train for (0 or less trains forever)")

    # VAE parameters
    parser.add_argument("--vae_model", type=str,
                        default="vae/models/seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data/",
                        help="Trained VAE model to load")
    parser.add_argument("--vae_model_type", type=str, default="cnn", help="VAE model type (\"cnn\" or \"mlp\")")
    parser.add_argument("--vae_z_dim", type=int, default="None", help="Size of VAE bottleneck")

    # Environment settings
    parser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
    parser.add_argument("--fps", type=int, default=30, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor")
    parser.add_argument("--switch_environment", type=int, default=250, help="How often to switch between environments")

    # Training parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See reward_functions.py for more info.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed " +
                             "with this option in our experience)")
    parser.add_argument("--eval_interval", type=int, default=5, help="Number of episodes between evaluation runs")
    parser.add_argument("--record_eval", type=bool, default=True,
                        help="If True, save videos of evaluation episodes " +
                             "to models/model_name/videos/")
    parser.add_argument("-restart", action="store_true",
                        help="If True, delete existing model in models/model_name before starting training")
    parser.add_argument(
            '-d', '--dot-extent',
            metavar='SIZE',
            default=2,
            type=int,
            help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    parser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    parser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    parser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    parser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    parser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    parser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    args = parser.parse_args()
    params = vars(parser.parse_args())
    restart = params["restart"]; del params["restart"]

    # Reset tf graph
    tf.compat.v1.reset_default_graph()

    # Start training
    train(args, restart)
