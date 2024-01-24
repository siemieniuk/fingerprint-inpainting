import numpy as np
import tensorflow as tf
import os
import time
import datetime
from IPython import display
from typing import Tuple, List
import csv


INPUT_SHAPE = (512, 384, 3)

def get_unet(
    depth: int = 3, dropout: float = 0.2, activation: str = "relu"
) -> tf.keras.Model:
    def get_downsample(x, n_filters: int, dropout: float = 0.2):
        conv_layer = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)
        conv_layer = tf.keras.layers.Dropout(dropout)(conv_layer)
        conv_layer = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(conv_layer)
        pool_layer = tf.keras.layers.MaxPooling2D((2, 2))(conv_layer)
        return conv_layer, pool_layer

    def get_upsample(x, conv_features, n_filters: int, dropout: float = 0.2):
        x = tf.keras.layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
        )(x)

        # fix dimensions
        out_pad = [[0, 0], [0, 0]]
        cf_shape_comp = np.array(conv_features.shape[1:])
        x_shape = np.array(x.shape[1:])
        if cf_shape_comp[1] != x_shape[1]:
            out_pad[1][0] = 1
        if cf_shape_comp[0] != x_shape[0]:
            out_pad[0][0] = 1
        x = tf.keras.layers.ZeroPadding2D(padding=out_pad)(x)

        # concatenation and other operations
        x = tf.keras.layers.Concatenate()([x, conv_features])
        x = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            activation=activation,
            kernel_initializer="he_normal",
        )(x)

        return x

    def get_bottleneck(x, n_filters: int, dropout: float = 0.2):
        bottleneck = tf.keras.layers.Conv2D(
            filters=current_filters,
            kernel_size=(3, 3),
            activation=activation,
            padding="same",
        )(x)
        bottleneck = tf.keras.layers.Dropout(0.1)(bottleneck)
        bottleneck = tf.keras.layers.Conv2D(
            filters=current_filters,
            kernel_size=(3, 3),
            activation=activation,
            padding="same",
        )(bottleneck)

        return bottleneck

    assert depth >= 1

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

    current_filters = 16
    c, p = get_downsample(inputs, current_filters, dropout=dropout)
    current_filters *= 2

    contractions = [c]
    pools = [p]

    # downsampling
    for i in range(depth - 1):
        c, p = get_downsample(pools[i], current_filters, dropout=dropout)
        contractions.append(c)
        pools.append(p)
        current_filters *= 2

    # bottleneck
    bottleneck = get_bottleneck(pools[-1], current_filters, dropout=dropout)

    # upsampling
    u = get_upsample(
        bottleneck, contractions[-1], current_filters, dropout=dropout
    )
    current_filters //= 2
    upsamples = [u]

    for i in range(depth - 1):
        u = get_upsample(
            upsamples[i],
            contractions[depth - i - 2],
            current_filters,
            dropout=dropout,
        )
        upsamples.append(u)
        current_filters //= 2

    # output
    outputs = tf.keras.layers.Conv2D(
        filters=1,  # monochromatic image
        kernel_size=(1, 1),
        padding="same",
        activation="sigmoid",
    )(upsamples[-1])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="unet")


def get_custom_conv(
    conv_activation: str = "relu",
    dense_activation: str = "relu"
):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100, activation=dense_activation),
        tf.keras.layers.Dense(400*275, activation="sigmoid"),
        tf.keras.layers.Reshape((400, 275, 1)),
        tf.keras.optimizers.RMSprop
    ])

    return model


def get_conv_only(conv_activation="relu"):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),

        tf.keras.layers.ZeroPadding2D(padding=(0, 288-275)),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=conv_activation),
        tf.keras.layers.UpSampling2D((2, 2)),

        tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding="same", activation="sigmoid"),
        tf.keras.layers.Cropping2D(((0, 0), (0, 21)))
    ])

    return model


def get_resnet_transfer() -> tf.keras.Model:
    def get_resnet_base(x):
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3))
        resnet.trainable = False
        return resnet(x)

    def get_upsample(x):
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Cropping2D(((13, 13), (4, 3)))(x)
        return x

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    inputs = tf.keras.layers.Resizing(224, 224)(inputs)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(inputs)

    resnet_base = get_resnet_base(preprocessed)
    upsamples = get_upsample(resnet_base)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(upsamples)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_mobilenet_transfer() -> tf.keras.Model:
    def get_mobilenet_base(x):
        mobilenet = tf.keras.applications.MobileNetV3Large(include_top=False, input_shape=(224, 224, 3))
        mobilenet.trainable = False
        return mobilenet(x)

    def get_upsample(x):
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = tf.keras.layers.UpSampling2D(size=(3, 2))(x)
        
        x = tf.keras.layers.Cropping2D(((19, 19), (9, 8)))(x)
        return x

    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    resized = tf.keras.layers.Resizing(224, 224)(inputs)
    preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(resized)

    mobilenet_base = get_mobilenet_base(preprocessed)
    upsamples = get_upsample(mobilenet_base)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(upsamples)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



class UnetGAN:
    def __init__(
            self, 
            generator_optimizer = None, discriminator_optimizer = None, 
            img_shape: Tuple[int, int, int] = (512, 384, 3),
            checkpoint_dir: str = './unetgan_training_checkpoints', log_dir: str = 'unetgan_logs/'
        ):
        self.img_shape = img_shape

        if generator_optimizer is None:
            self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        else:
            self.generator_optimizer = generator_optimizer

        if discriminator_optimizer is None:
            self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        else:
            self.discriminator_optimizer = discriminator_optimizer

        self.OUTPUT_CHANNELS = 1
        self.LAMBDA = 100
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = []

        # Instantiating stuff
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

        self.summary_writer = tf.summary.create_file_writer(
            log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        self.gen_total_loss = tf.keras.metrics.Mean(name='gen_total_loss')
        self.gen_gan_loss = tf.keras.metrics.Mean(name='gen_gan_loss')
        self.gen_l1_loss = tf.keras.metrics.Mean(name='gen_l1_loss')
        self.disc_loss = tf.keras.metrics.Mean(name='disc_loss')


    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=self.img_shape)

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            self.downsample(128, 4),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(512, 4),  # (batch_size, 8, 8, 512)
            self.downsample(512, 4),  # (batch_size, 4, 4, 512)
            self.downsample(512, 4),  # (batch_size, 2, 2, 512)
            # self.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            # self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='sigmoid')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss


    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=self.img_shape, name='input_image')
        tar = tf.keras.layers.Input(shape=(self.img_shape[0], self.img_shape[1], self.OUTPUT_CHANNELS), name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


    # def generate_images(model, test_input, tar):
    #     prediction = model(test_input, training=True)
    #     plt.figure(figsize=(15, 15))

    #     display_list = [test_input[0], tar[0], prediction[0]]
    #     title = ['Input Image', 'Ground Truth', 'Predicted Image']

    #     for i in range(3):
    #         plt.subplot(1, 3, i+1)
    #         plt.title(title[i])
    #         # Getting the pixel values in the [0, 1] range to plot.
    #         plt.imshow(display_list[i] * 0.5 + 0.5)
    #         plt.axis('off')
    #     plt.show()


    def log_to_csv(self, filename, fieldnames, data):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow(data)

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        self.gen_total_loss.update_state(gen_total_loss)
        self.gen_gan_loss.update_state(gen_gan_loss)
        self.gen_l1_loss.update_state(gen_l1_loss)
        self.disc_loss.update_state(disc_loss)

        for metric in self.metrics:
            metric.update_state(target, gen_output)

        # with self.summary_writer.as_default():
        #     tf.summary.scalar('gen_total_loss', self.gen_total_loss.result(), step=step)
        #     tf.summary.scalar('gen_gan_loss', self.gen_gan_loss.result(), step=step)
        #     tf.summary.scalar('gen_l1_loss', self.gen_l1_loss.result(), step=step)
        #     tf.summary.scalar('disc_loss', self.disc_loss.result(), step=step)


    def fit(self, train_ds, valid_ds, epochs, metrics: List[tf.keras.metrics.Metric] = [], checkpoint_steps=4000, log_dir="logs"):
        self.metrics = metrics
        if os.path.isdir(log_dir):
            os.rmdir(log_dir)
            os.mkdir(log_dir)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        history = {
            'gen_total_loss': [],
            'gen_gan_loss': [],
            'gen_l1_loss': [],
            'disc_loss': [],
        }
        for metric in self.metrics:
            history[f"train_{metric.name}"] = []
            history[f"val_{metric.name}"] = []

        # Training loop
        start = time.time()
        for epoch in range(epochs):
            display.clear_output(wait=True)
            if epoch != 0:
                print(f'Time taken for 1 epoch: {time.time()-start:.2f} sec\n')
            start = time.time()
            # generate_images(generator, example_input, example_target)
            print(f"History:\n{history}")
            print(f"Epoch: {epoch}")

            epoch_losses = {
                'gen_total_loss': tf.keras.metrics.Mean(),
                'gen_gan_loss': tf.keras.metrics.Mean(),
                'gen_l1_loss': tf.keras.metrics.Mean(),
                'disc_loss': tf.keras.metrics.Mean(),
            }
            for step, (input_image, target) in train_ds.enumerate():
                self.train_step(input_image, target)

                # Training step
                if (step+1) % 100 == 0:
                    print('.', end='', flush=True)

                self.log_to_csv(
                    f"{log_dir}/losses.csv", 
                    ["epoch", "step", "gen_total_loss", 'gen_gan_loss', 'gen_l1_loss', 'disc_loss'], 
                    {
                        "epoch": epoch,
                        "step": step.numpy(),
                        "gen_total_loss": self.gen_total_loss.result().numpy(),
                        'gen_gan_loss': self.gen_gan_loss.result().numpy(),
                        'gen_l1_loss': self.gen_l1_loss.result().numpy(),
                        'disc_loss': self.disc_loss.result().numpy()
                    }
                )

                epoch_losses['gen_total_loss'](self.gen_total_loss.result())
                epoch_losses['gen_gan_loss'](self.gen_gan_loss.result())
                epoch_losses['gen_l1_loss'](self.gen_l1_loss.result())
                epoch_losses['disc_loss'](self.disc_loss.result())

                self.gen_total_loss.reset_state()
                self.gen_gan_loss.reset_state()
                self.gen_l1_loss.reset_state()
                self.disc_loss.reset_state()

                if (step+1) % checkpoint_steps == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                
            history['gen_total_loss'].append(epoch_losses['gen_total_loss'].result().numpy())
            history['gen_gan_loss'].append(epoch_losses['gen_gan_loss'].result().numpy())
            history['gen_l1_loss'].append(epoch_losses['gen_l1_loss'].result().numpy())
            history['disc_loss'].append(epoch_losses['disc_loss'].result().numpy())

            metrics_dict = {"epoch": epoch}
            # Metrics logging and adding to 'history'
            for metric in self.metrics:
                history[f"train_{metric.name}"].append(metric.result().numpy())
                metrics_dict[f"train_{metric.name}"] = metric.result().numpy()
                metric.reset_state()

            for x_batch_val, y_batch_val in valid_ds:
                pred_batch_val = self.generator(x_batch_val, training=False)
                for metric in self.metrics:
                    metric.update_state(y_batch_val, pred_batch_val)
            for metric in self.metrics:
                history[f"val_{metric.name}"].append(metric.result().numpy())
                metrics_dict[f"val_{metric.name}"] = metric.result().numpy()
                metric.reset_state()
            self.log_to_csv(f"{log_dir}/metrics.csv", list(metrics_dict.keys()), metrics_dict)
            
        self.metrics = []

        return history