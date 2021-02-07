import gin
import tensorflow as tf
import logging
import datetime


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval,
                 ckpt_interval, problem_type):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.problem_type = problem_type
        self.max_acc = 0
        self.min_loss = 100

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Summary Writer
        # ....
        self.summary_path_train = self.run_paths['path_summary_train']
        self.summary_path_val = self.run_paths['path_summary_val']
        self.train_summary_writer = tf.summary.create_file_writer(self.summary_path_train)
        self.val_summary_writer = tf.summary.create_file_writer(self.summary_path_val)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Loss objective
        if self.problem_type == 'classification':
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        elif self.problem_type == 'regression':
            self.loss_object = tf.keras.losses.Huber(delta=0.3)
            self.train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
            self.val_accuracy = tf.keras.metrics.Accuracy(name='val_accuracy')

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)

        # Checkpoint Manager
        # ...
        print("current_time")
        print(self.current_time)
        # self.checkpoint_path = './checkpoint/train/' + self.current_time
        self.checkpoint_path = self.run_paths["path_ckpts_train"]

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

    @tf.function
    def train_step(self, images, labels):

        # label_preds = np.argmax(predictions, -1)
        # label=labels.numpy()
        # binary_true=np.squeeze(labels)
        # binary_pred=np.squeeze(label_preds)

        # binary_accuracy = metrics.accuracy_score(binary_true, binary_pred)
        # binary_confusion_matrix = metrics.confusion_matrix(binary_true, binary_pred)

        # tf.print(binary_accuracy)
        # tf.print(binary_confusion_matrix)

        if self.problem_type == 'regression':
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = self.model(images, training=True)
                loss = self.loss_object(labels, predictions)
                predictions = tf.cast(tf.clip_by_value(predictions + 0.5, clip_value_min=0, clip_value_max=4), tf.int32)

        elif self.problem_type == 'classification':
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = self.loss_object(labels, predictions)

        else:
            raise ValueError
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        v_loss = self.loss_object(labels, predictions)

        # label_preds = np.argmax(predictions, -1)
        # labels = labels.numpy()
        # binary_true = np.squeeze(labels)
        # binary_pred = np.squeeze(label_preds)
        #
        # binary_accuracy = metrics.accuracy_score(binary_true, binary_pred)
        # binary_confusion_matrix = metrics.confusion_matrix(binary_true, binary_pred)
        #
        # tf.print(binary_accuracy)
        # tf.print(binary_confusion_matrix)

        self.val_loss(v_loss)
        if self.problem_type == 'regression':
            predictions = tf.cast(tf.clip_by_value(predictions + 0.5, clip_value_min=0, clip_value_max=4), tf.int32)
            self.val_loss(v_loss)
            self.val_accuracy(labels, predictions)
        elif self.problem_type == 'classification':
            self.val_loss(v_loss)
            self.val_accuracy(labels, predictions)
        else:
            raise ValueError

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                # print(step)

                # Reset validation metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Train Loss: {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                # ...train test loss accuracy
                with self.train_summary_writer.as_default():
                    # tf.summary.scalar('train_loss', self.train_loss.result(), step=step)
                    # tf.summary.scalar('train_accuracy', self.train_accuracy.result() * 100, step=step)
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result() * 100, step=step)

                with self.val_summary_writer.as_default():
                    # tf.summary.scalar('val_loss', self.val_loss.result(), step=step)
                    # tf.summary.scalar('val_accuracy', self.val_accuracy.result() * 100, step=step)
                    tf.summary.scalar('loss', self.val_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.val_accuracy.result() * 100, step=step)

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
                    # else:
                        # print("Validation loss is not better, no new checkpoint")

                # Nothing happens
                # else:
                    # print("Validation loss is not better, no new checkpoint")

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                # ...
                # save_path = self.ckpt_manager.save()
                # print("Saved checkpoint for final step: {}".format(save_path))
                logging.info("best validation loss {:1.2f}".format(self.min_loss))
                logging.info("the accuracy {:1.2f}".format(self.max_acc))

                return self.val_accuracy.result().numpy()
