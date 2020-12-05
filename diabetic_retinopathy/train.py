import gin
import tensorflow as tf
import logging


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval):
        # Summary Writer
        # ....
        self.train_loss_summary_writer = tf.summary.create_file_writer("./train_loss")
        self.train_accuracy_summary_writer = tf.summary.create_file_writer("./train_accuracy")
        self.test_loss_summary_writer = tf.summary.create_file_writer("./test_loss")
        self.test_accuracy_summary_writer = tf.summary.create_file_writer("./test_accuracy")

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Checkpoint Manager
        # ...
        self.checkpoint_path = './checkpoint/train'
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                print(step)

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Write summary to tensorboard
                # ...train test loss accuracy
                with self.train_loss_summary_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                    # tf.summary.scalar('train_loss', self.train_loss.result(), step=self.optimizer.iterations)

                with self.train_accuracy_summary_writer.as_default():
                    tf.summary.scalar('train_accuracy', self.train_accuracy.result() * 100, step=step)
                    # tf.summary.scalar('train_accuracy', self.train_accuracy.result() * 100,
                    #                  step=self.optimizer.iterations)

                with self.test_loss_summary_writer.as_default():
                    tf.summary.scalar('test_loss', self.test_loss.result(), step=step)

                with self.test_accuracy_summary_writer.as_default():
                    tf.summary.scalar('test_accuracy', self.test_accuracy.result() * 100, step=step)

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                # ...
                save_path = self.ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(step), save_path))
                print("loss {:1.2f}".format(self.train_loss.result()))
                print("accuracy {:1.2f}".format(self.train_accuracy.result()))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                # ...
                save_path = self.ckpt_manager.save()
                print("Saved checkpoint for final step: {}".format(save_path))
                print("loss {:1.2f}".format(self.train_loss.result().numpy()))
                print("accuracy {:1.2f}".format(self.train_accuracy.result().numpy()))

                return self.test_accuracy.result().numpy()
