import gin
import tensorflow as tf
import logging
import datetime


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval,
                 ckpt_interval):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.max_acc = 0
        self.min_loss = 100

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Summary Writer
        # ....
        self.summary_path = self.run_paths['path_summary']
        self.train_summary_writer = tf.summary.create_file_writer(self.summary_path + self.current_time + 'train')
        self.test_summary_writer = tf.summary.create_file_writer(self.summary_path + self.current_time + 'train')

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Checkpoint Manager
        # ...
        print("current_time")
        print(self.current_time)
        # self.checkpoint_path = './checkpoint/train/' + self.current_time
        self.checkpoint_path = self.run_paths["path_ckpts_train"]
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(features, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, features, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(features, training=False)
        v_loss = self.loss_object(labels, predictions)

        self.val_loss(v_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (feature, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(feature, labels)

            if step % self.log_interval == 0:

                # Reset validation metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_feature, val_labels in self.ds_val:
                    self.val_step(val_feature, val_labels)

                template = 'Step {}, Train Loss: {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                # ...train test loss accuracy
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('train_accuracy', self.train_accuracy.result() * 100, step=step)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('val_accuracy', self.val_accuracy.result() * 100, step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Compare it with max_acc
                acc = self.val_accuracy.result().numpy()
                loss = self.val_loss.result().numpy()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:

                # Check if val_loss decrease or not
                if self.max_acc < acc:
                    self.min_loss = loss
                    self.max_acc = acc
                    logging.info(f'Saving better checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    print("loss {:1.2f}".format(loss))
                    # Save checkpoint
                    # ...
                    save_path = self.ckpt_manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(step), save_path))

                elif self.val_accuracy == acc:
                    if self.val_loss < loss:
                        self.min_loss = loss
                        self.max_acc = acc
                        logging.info(f'Saving better checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        print("validation loss {:1.2f}".format(loss))
                        # Save checkpoint
                        # ...
                        self.ckpt_manager.save()

                    # Nothing happens
                #     else:
                #         print("Validation loss is not better, no new checkpoint")

                # # Nothing happens
                # else:
                #     print("Validation loss is not better, no new checkpoint")

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                # ...
                # save_path = self.ckpt_manager.save()
                # print("Saved checkpoint for final step: {}".format(save_path))
                logging.info("best validation loss {:1.2f}".format(self.min_loss))
                logging.info("the accuracy {:1.2f}".format(self.max_acc))

                return self.val_accuracy.result().numpy()
