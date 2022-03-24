import argparse
import gzip
import os
import pickle
import shutil
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize

#take the first frame
def preprocess_depth_frame(frame):
    frame = frame[:, :, :1]                 # RGBA -> R
    return frame

#normalise the final frame
def interlaced(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame

#creates a new stack of images
def normalise(images):
    normalised = []
    i = 0
    counter = len(images)
    while i < counter:
        arr = images[i]
        arr = arr.astype(np.float32) / 255.0
        normalised.append(arr)
        i+=1

    return np.stack(normalised, axis=0)

#preprocess YUV frame
def preprocess_UV(frame):
    frame = frame[:, :, 1:3]
    #frame = frame.astype(np.float32) / 12.0 # [0, 12=num_classes] -> [0, 1]
    return frame

#prepocess YUV frame
def preprocess_Y(frame):
    frame = frame[:, :, :1]                 # RGBA -> R
    #frame = frame.astype(np.float32) / 12.0 # [0, 12=num_classes] -> [0, 1]
    return frame

#convert colour space of YUV
def RGB2YUV(rgb):

    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv.astype(int)

##convert segmentation to YUV
def toYUV_preprocess(image):
    stepOne = RGB2YUV(image)
    stepTwo = preprocess_UV(stepOne)
    return stepTwo

#combine logarithmic depth image and UV components
def final_representation(Y_images, U_V_images):
    images = []
    i = 0
    counter = len(Y_images)
    while i < counter:
        arr = np.dstack((Y_images[i], U_V_images[i]))
        arr = arr.astype(np.float32) / 255
        images.append(arr)
        i+=1

    return np.stack(images, axis=0)

#load iamges
def load_images(dir_path, preprocess_fn):
    images = []
    for filename in os.listdir(dir_path):
        _, ext = os.path.splitext(filename)
        if ext == ".png":
            filepath = os.path.join(dir_path, filename)
            frame = preprocess_fn(np.asarray(Image.open(filepath)))
            images.append(frame)
    return np.stack(images, axis=0)

#train validation split
def train_val_split(images, val_portion=0.1):
    val_split = int(images.shape[0] * val_portion)
    train_images = images[val_split:]
    val_images = images[:val_split]
    return train_images, val_images

#loss function
def kl_divergence(mean, logstd_sq, name="kl_divergence"):
    with tf.variable_scope(name):
        return -0.5 * tf.reduce_sum(1.0 + logstd_sq - tf.square(mean) - tf.exp(logstd_sq), axis=1)

#loss function
def bce_loss(labels, logits, targets):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )

#loss function
def bce_loss_v2(labels, logits, targets, epsilon=1e-10):
    with tf.variable_scope("bce"):
        return -(labels * tf.log(epsilon + targets) + (1 - labels) * tf.log(epsilon + 1 - targets))

#verify range
def verify_range(tensor, vmin, vmax):
    verify_op = tf.Assert(tf.reduce_all(tf.logical_and(tensor >= vmin, tensor <= vmax)),
                                ["min=", tf.reduce_min(tensor), "max=", tf.reduce_max(tensor)],
                                name="verify_range")
    with tf.control_dependencies([verify_op]):
        tensor = tf.multiply(tensor, 1, name="verify_tensor_identity")
    return tensor

#check size for data augmentation
def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

#resize function
def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image

#randomly rotate
def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image

#randommly rotate an image
def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random'):
    image = np.copy(image_origin)
    if np.random.rand() > p:
        return image
    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image

#blur image
def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)

#augment the images in a routine and stack
def augment_images(images):
    aug_images = []
    counter = len(images)
    aug_counter = 0
    i = 0
    while i < counter:
        if aug_counter == 0:
            newImage = random_rotation(images[i])
            aug_counter = 1

        elif aug_counter == 1:
            newImage = random_erasing(images[i])
            aug_counter = 2

        elif aug_counter == 2:
            newImage = np.flipud(images[i])
            aug_counter = 3

        elif aug_counter == 3:
            newImage = np.fliplr(images[i])
            aug_counter = 4

        elif aug_counter == 4:
            newImage = blur_image(images[i])
            aug_counter = 0
        else:
            break
        aug_images.append(newImage)
        i+=1

    return np.stack(aug_images, axis=0)


#vae class taken from the project reference in the report
class VAE():
    """
        Base variational autoencoder class.
    """

    def __init__(self, source_shape, target_shape, build_encoder_fn, build_decoder_fn,
                 z_dim=512, beta=1.0, learning_rate=1e-4, lr_decay=0.98, kl_tolerance=0.0,
                 model_dir=".", loss_fn=bce_loss, training=True, reuse=tf.compat.v1.AUTO_REUSE,
                 **kwargs):
        # Create vae
        self.source_shape = source_shape
        self.target_shape = target_shape
        self.z_dim = z_dim
        self.beta = beta
        self.kl_tolerance = kl_tolerance
        with tf.compat.v1.variable_scope("vae", reuse=reuse):
            # Get and verify source and target
            self.source_states = tf.compat.v1.placeholder(shape=(None, *self.source_shape), dtype=tf.float32, name="source_state_placeholder")
            self.target_states = tf.compat.v1.placeholder(shape=(None, *self.target_shape), dtype=tf.float32, name="target_state_placeholder")
            source_states = verify_range(self.source_states, vmin=0, vmax=1)
            target_states = verify_range(self.target_states, vmin=0, vmax=1)

            # Encode image
            with tf.variable_scope("encoder", reuse=False):
                encoded = build_encoder_fn(source_states)

            # Get encoded mean and std
            self.mean      = tf.layers.dense(encoded, z_dim, activation=None, name="mean")
            self.logstd_sq = tf.layers.dense(encoded, z_dim, activation=None, name="logstd_sqare")

            # Sample normal distribution
            self.normal = tfp.distributions.Normal(self.mean, tf.exp(0.5 * self.logstd_sq), validate_args=True)
            if training:
                self.sample = tf.squeeze(self.normal.sample(1), axis=0)
            else:
                self.sample = self.mean

            # Decode random sample
            with tf.variable_scope("decoder", reuse=False):
                decoded = build_decoder_fn(self.sample)

            # Reconstruct image
            self.reconstructed_logits = tf.layers.flatten(decoded, name="reconstructed_logits")
            self.reconstructed_states = tf.nn.sigmoid(self.reconstructed_logits, name="reconstructed_states")

            # Epoch variable
            self.step_idx = tf.Variable(0, name="step_idx", trainable=False)
            self.inc_step_idx = tf.compat.v1.assign(self.step_idx, self.step_idx + 1)

            # Create optimizer
            if training:
                # Reconstruction loss
                self.flattened_target = tf.layers.flatten(target_states, name="flattened_target")
                self.reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        loss_fn(labels=self.flattened_target, logits=self.reconstructed_logits, targets=self.reconstructed_states),
                        axis=1
                    )
                )

                # KL divergence loss
                self.kl_loss = kl_divergence(self.mean, self.logstd_sq, name="kl_divergence")
                if self.kl_tolerance > 0:
                    self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_dim)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                # Total loss
                self.loss = self.reconstruction_loss + self.beta * self.kl_loss

                # Create optimizer
                self.learning_rate = tf.train.exponential_decay(learning_rate, self.step_idx, 1, lr_decay, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_step = self.optimizer.minimize(self.loss)

                # Summary
                self.mean_kl_loss, self.update_mean_kl_loss = tf.metrics.mean(self.kl_loss)
                self.mean_reconstruction_loss, self.update_mean_reconstruction_loss = tf.metrics.mean(self.reconstruction_loss)
                self.merge_summary = tf.summary.merge([
                    tf.summary.scalar("kl_loss", self.mean_kl_loss),
                    tf.summary.scalar("reconstruction_loss", self.mean_reconstruction_loss),
                    tf.summary.scalar("learning_rate", self.learning_rate)
                ])

            # Setup model saver and dirs
            self.saver = tf.compat.v1.train.Saver()
            self.model_dir = model_dir
            self.checkpoint_dir = "{}/checkpoints/".format(self.model_dir)
            self.log_dir        = "{}/logs/".format(self.model_dir)
            self.dirs = [self.checkpoint_dir, self.log_dir]
            for d in self.dirs: os.makedirs(d, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        if init_logging:
            self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"), self.sess.graph)
            self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "val"), self.sess.graph)

    def save(self):
        model_checkpoint = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(self.sess, model_checkpoint, global_step=self.step_idx)
        print("Model checkpoint saved to {}".format(model_checkpoint))

    def load_latest_checkpoint(self):
        model_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if model_checkpoint:
            try:
                self.saver.restore(self.sess, model_checkpoint)
                print("Model checkpoint restored from {}".format(model_checkpoint))
                return True
            except Exception as e:
                print(e)
                return False

    def generate_from_latent(self, z):
        return self.sess.run(self.reconstructed_states, feed_dict={
                self.sample: z
            })

    def reconstruct(self, source_states):
        reconstructed_states = self.sess.run(self.reconstructed_states, feed_dict={
                self.source_states: source_states
            })
        return [s.reshape(self.source_shape) for s in reconstructed_states]

    def encode(self, source_states):
        return self.sess.run(self.mean, feed_dict={
                self.source_states: source_states
            })

    def get_step_idx(self):
        return tf.train.global_step(self.sess, self.step_idx)

    def train_one_epoch(self, train_source, train_target, batch_size):
        indices = np.arange(len(train_source))
        np.random.shuffle(indices)
        self.sess.run(tf.local_variables_initializer())
        for i in range(train_source.shape[0] // batch_size):
            mb_idx = indices[i*batch_size:(i+1)*batch_size]
            self.sess.run([self.train_step, self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.source_states: train_source[mb_idx],
                self.target_states: train_target[mb_idx]
            })
        self.train_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        self.sess.run(self.inc_step_idx)

    def evaluate(self, val_source, val_target, batch_size):
        indices = np.arange(len(val_source))
        np.random.shuffle(indices)
        self.sess.run(tf.local_variables_initializer())
        for i in range(val_source.shape[0] // batch_size):
            mb_idx = indices[i*batch_size:(i+1)*batch_size]
            self.sess.run([self.update_mean_kl_loss, self.update_mean_reconstruction_loss], feed_dict={
                self.source_states: val_source[mb_idx],
                #print(val_source[mb_idx])
                self.target_states: val_target[mb_idx],
                #print(val_target[mb_idx])
            })
        self.val_writer.add_summary(self.sess.run(self.merge_summary), self.get_step_idx())
        return self.sess.run([self.mean_reconstruction_loss, self.mean_kl_loss])

#convolutional base class taken from project reference in the report
class ConvVAE(VAE):
    """
        Convolutional VAE class.
        Achitecture from: https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py
    """

    def __init__(self, source_shape, target_shape=None, **kwargs):
        target_shape = source_shape if target_shape is None else target_shape

        def build_encoder(x):

            x = tf.layers.conv2d(x, filters=32,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv1")
            x = tf.layers.conv2d(x, filters=64,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv2")
            x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv3")
            x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv4")
            self.encoded_shape = x.shape[1:]
            #print(self.encoded_shape)
            x = tf.layers.flatten(x, name="flatten")
            return x

        def build_decoder(z):
            x = tf.layers.dense(z, np.prod(self.encoded_shape), activation=None, name="dense1")
            x = tf.reshape(x, (-1, *self.encoded_shape))
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="deconv1")
            x = tf.layers.conv2d_transpose(x, filters=64,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="deconv2")
            x = tf.layers.conv2d_transpose(x, filters=32,  kernel_size=5, strides=2, activation=tf.nn.relu, padding="valid", name="deconv3")
            x = tf.layers.conv2d_transpose(x, filters=target_shape[-1], kernel_size=4, strides=2, activation=None, padding="valid", name="deconv4")
            #assert x.shape[1:] == target_shape, f"{x.shape[1:]} != {target_shape}"
            return x

        super().__init__(source_shape, target_shape, build_encoder, build_decoder, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a VAE with RGB images as source and RGB or segmentation images as target")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="images")
    parser.add_argument("--use_segmentation_as_target", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--kl_tolerance", type=float, default=0.0)
    parser.add_argument("-restart", action="store_true")
    args = parser.parse_args()

    # Load images from dataset/rgb and dataset/segmentation folders for the 5 towns

    RGB_images = load_images(os.path.join("Town01_images", "Interlaced"), toYUV_preprocess)
    logDepth_images = load_images(os.path.join("Town01_images", "logDepth"), preprocess_depth_frame)

    RGB_images2 = load_images(os.path.join("Town02_images", "Interlaced"), toYUV_preprocess)
    logDepth_images2 = load_images(os.path.join("Town02_images", "logDepth"), preprocess_depth_frame)

    RGB_images3 = load_images(os.path.join("Town03_images", "Interlaced"), toYUV_preprocess)
    logDepth_images3 = load_images(os.path.join("Town03_images", "logDepth"), preprocess_depth_frame)

    RGB_images4 = load_images(os.path.join("Town04_images", "Interlaced"), toYUV_preprocess)
    logDepth_images4 = load_images(os.path.join("Town04_images", "logDepth"), preprocess_depth_frame)

    RGB_images5 = load_images(os.path.join("Town05_images", "Interlaced"), toYUV_preprocess)
    logDepth_images5 = load_images(os.path.join("Town05_images", "logDepth"), preprocess_depth_frame)

#combine and convert representations
    town1 = final_representation(logDepth_images, RGB_images)
    aug_town1 = augment_images(town1)

    town2 = final_representation(logDepth_images2, RGB_images2)
    aug_town2 = augment_images(town2)

    town3 = final_representation(logDepth_images3, RGB_images3)
    aug_town3 = augment_images(town3)

    town4 = final_representation(logDepth_images4, RGB_images4)
    aug_town4 = augment_images(town4)

    town5 = final_representation(logDepth_images5, RGB_images5)
    aug_town5 = augment_images(town5)

    imagesT = (town1, aug_town1, town2, aug_town2, town3, aug_town3, town4, aug_town4, town5, aug_town5)
    images = np.vstack(imagesT)

    images = normalise(images)

    #Split into train and vl sets
    np.random.seed(0)
    train_source_images, val_source_images = train_val_split(images, val_portion=0.2)

    train_target_images, val_target_images = train_source_images, val_source_images

    # Get source and target image sizes
    # (may be different e.g. RGB and grayscale)
    source_shape = train_source_images.shape[1:]
    target_shape = train_target_images.shape[1:] if args.use_segmentation_as_target else source_shape

    # Set model name from params
    if args.model_name is None:
        args.model_name = "{}_{}_{}_zdim{}_beta{}_kl_tolerance{}_{}".format(
            "seg" if args.use_segmentation_as_target else "rgb",
            args.loss_type, args.model_type, args.z_dim, args.beta, args.kl_tolerance,
            os.path.splitext(os.path.basename(args.dataset))[0])

    print("train_source_images.shape", train_source_images.shape)
    print("val_source_images.shape", val_source_images.shape)
    print("train_target_images.shape", train_target_images.shape)
    print("val_target_images.shape", val_target_images.shape)
    print("")
    print("Training parameters:")
    for k, v, in vars(args).items(): print(f"  {k}: {v}")
    print("")

    if args.loss_type == "bce": loss_fn = bce_loss
    else: raise Exception("No loss function \"{}\"".format(args.loss_type))

    if args.model_type == "cnn": VAEClass = ConvVAE
    else: raise Exception("No model type \"{}\"".format(args.model_type))

    # Create VAE model
    vae = VAEClass(source_shape=source_shape,
                   target_shape=target_shape,
                   z_dim=args.z_dim,
                   beta=args.beta,
                   learning_rate=args.learning_rate,
                   lr_decay=args.lr_decay,
                   kl_tolerance=args.kl_tolerance,
                   loss_fn=loss_fn,
                   model_dir=os.path.join("models", args.model_name))

    # Prompt to load existing model if any
    if not args.restart:
        if os.path.isdir(vae.log_dir) and len(os.listdir(vae.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(args.model_name))
            if answer.upper() == "C":
                pass
            elif answer.upper() == "R":
                args.restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(args.model_name))

    if args.restart:
        shutil.rmtree(vae.model_dir)
        for d in vae.dirs:
            os.makedirs(d)
    vae.init_session()
    if not args.restart:
        vae.load_latest_checkpoint()

    # Training loop
    min_val_loss = float("inf")
    counter = 0
    print("Training")
    while True:
        epoch = vae.get_step_idx()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}")

        # Calculate evaluation metrics
        val_loss, _ = vae.evaluate(val_source_images, val_target_images, args.batch_size)

        # Early stopping
        if val_loss < min_val_loss:
            counter = 0
            min_val_loss = val_loss
            vae.save() # Save if better
        else:
            counter += 1
            if counter >= 10:
                print("No improvement in last 10 epochs, stopping")
                break

        # Train one epoch
        vae.train_one_epoch(train_source_images, train_target_images, args.batch_size)
