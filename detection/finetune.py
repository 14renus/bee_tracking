import os
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from . import train_detection
from . import unet
from utils import func
from utils.paths import DET_DATA_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES

class FinetuneModel(train_detection.TrainModel):
    '''
    Loads graph from saved .meta file, without recreating graph. Then adds on finetuneable layers.
    '''

    def __init__(self, data_path, train_prop, with_augmentation, random_frame_tiles, dropout_ratio=0, learning_rate=train_detection.BASE_LR,
                 loss_upweight=10, angle_loss_weight=1, set_random_seed=False, num_classes=3, continue_finetuning_from_checkpoint=False):
        super(FinetuneModel, self).__init__(data_path, train_prop, with_augmentation, random_frame_tiles,
                                            dropout_ratio, learning_rate, loss_upweight, angle_loss_weight, set_random_seed, num_classes)
        self.continue_finetuning = continue_finetuning_from_checkpoint


    def _loss(self, logits, label, weight, angle_pred, angle_label, prior):
        last_relu = prior

        loss_softmax = unet.loss(logits, label, weight, self.num_classes)
        loss_angle = unet.angle_loss(angle_pred, angle_label, weight, ignore_bg=True, use_weights=False)

        total_loss = loss_softmax + self.angle_loss_weight * loss_angle #tf.add_n(losses, name='total_loss')
        return logits, total_loss, last_relu, angle_pred, loss_softmax, loss_angle

    def build_model(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.acc_file = os.path.join(checkpoint_dir, "accuracy.csv")
        cpu, gpu = func.find_devices()
        tf_dev = gpu if gpu != "" else cpu

        with tf.Graph().as_default() as graph, tf.device(cpu):
            if self.set_random_seed:
                tf.set_random_seed(train_detection.SEED)

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            opt = tf.train.AdamOptimizer(learning_rate=train_detection.BASE_LR)
            self.is_train = tf.placeholder(tf.bool, shape=[])
            self.placeholder_img = tf.placeholder(tf.float32, shape=(None, DS, DS, 1), name="images")
            self.placeholder_label = tf.placeholder(tf.uint8, shape=(None, DS, DS), name="labels")
            self.placeholder_weight = tf.placeholder(tf.float32, shape=(None, DS, DS), name="weight")
            self.placeholder_angle_label = tf.placeholder(tf.float32, shape=(None, DS, DS), name="angle_labels")
            self.placeholder_prior = tf.placeholder(tf.float32, shape=(None, DS, DS, NUM_FILTERS), name="prior")

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)):
                # Create original network.
                logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, self.placeholder_img,
                                                                  self.is_train, prev=self.placeholder_prior,
                                                                  num_classes=CLASSES)

            if not self.continue_finetuning:
                self.saver = tf.train.Saver(tf.global_variables())

            # Add new finetuned layer.
            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                conv_logits = unet._create_conv_relu(last_relu, "new_conv_logits", NUM_FILTERS,
                                                     dropout_ratio=0,
                                                     is_training=self.is_train)
                classes = self.num_classes if self.num_classes > 2 else 1
                logits = unet._create_conv_relu(conv_logits, "new_logits", classes, 0,
                                                is_training=self.is_train)
                logits, total_loss, last_relu, angle_pred, loss_softmax, loss_angle = self._loss(logits, self.placeholder_label, self.placeholder_weight,
                                                                 angle_pred, self.placeholder_angle_label, last_relu)
                self.outputs = (logits, total_loss, last_relu, angle_pred, loss_softmax, loss_angle)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                grads = opt.compute_gradients(total_loss)

            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            variable_averages = tf.train.ExponentialMovingAverage(func.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            batchnorm_updates_op = tf.group(*update_ops)
            self.train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

            if self.continue_finetuning:
                self.saver = tf.train.Saver(tf.global_variables())

            checkpoint = func.find_last_checkpoint(checkpoint_dir)
            print("Restoring checkpoint %i.. from %s" % (checkpoint, checkpoint_dir), flush=True)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))
            checkpoint += 1

            # (Re)init saver to include added global variables.
            self.saver = tf.train.Saver(tf.global_variables())

            # Init added variables only, not variables loaded from checkpoint.
            global_vars = tf.global_variables()
            is_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]
            if len(not_initialized_vars):
                self.sess.run(tf.variables_initializer(not_initialized_vars))
            self.sess.run(tf.local_variables_initializer())
        return checkpoint


def run_finetuning(data_path=DET_DATA_DIR, checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2"),
                   output_checkpoint_dir=None,
                   train_prop=0.9, n_iters=10, with_augmentation=True, random_frame_tiles=False,
                   dropout_ratio=0,
                   learning_rate=train_detection.BASE_LR,
                   loss_upweight=10, angle_loss_weight=0,
                   set_random_seed=False,
                   num_classes=CLASSES, return_img=False,
                   continue_finetuning_from_saved_checkpoint=False):
    '''
    Run train and test iterations on unet2 for n_iters.

    Saves mettrics and checkpoints to checkpoint_dir.

    :param data_path: dir holding .npz files.
    :param checkpoint_dir: used to build latest checkpoint, also to store newly trained checkpoints only if output_checkpoint_dir is not None.
    :param output_checkpoint_dir: if not None used to store newly trained checkpoints.
    :param train_prop: proportion of each .npz file to be trained on, rest is reserved for test.
    :param n_iters: how many .npz files to iterate through.
    :param with_augmentation: whether to randomly flip horizontally and vertically (train data only).
    :param random_frame_tiles: whether to randomly choose BATCH_SIZE number of DS x DS tiles within a frame to train on.
                               defaults to choosing first four tiles of DS x DS.
    :param loss_upweight: positive weight to upweight pixels when calculating average loss
    :param angle_loss_weight: weight of angle_loss in total_loss calculation (total_loss = softmax_loss + angle_loss_weight * angle_loss)
    :param set_random_seed: tries to remove variation amongst training runs.
    :param return_img: whether to return segmentation and angle preds on test images
    :param continue_finetuning_from_saved_checkpoint: whether to skip adding new nodes, and just continiue training
    :return:
      model_obj: TrainModel object.
      img: if return_img is true, return last iteration's predictions on test (list of tuples of segmentation & angle preds)
      iters: total number of iterations performed to train model_obj (picks up from last checkpoint)
    '''
    model_obj = FinetuneModel(data_path, train_prop, with_augmentation, random_frame_tiles, dropout_ratio, learning_rate, loss_upweight, angle_loss_weight, set_random_seed, num_classes, continue_finetuning_from_saved_checkpoint)
    start_iter = model_obj.build_model(checkpoint_dir)
    if output_checkpoint_dir:
        func.make_dir(output_checkpoint_dir)
        model_obj.checkpoint_dir = output_checkpoint_dir
    return train_detection.run_training_on_model(model_obj, start_iter, n_iters, return_img)