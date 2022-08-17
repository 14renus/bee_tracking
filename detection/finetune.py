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

    def _loss(self, img, label, weight, angle_label, prior, graph, scope):
        last_relu = graph.get_tensor_by_name(scope + "Relu_13:0")
        angle_pred = graph.get_tensor_by_name(scope + "angle_pred_bn/cond/Merge:0")
        conv_logits = unet._create_conv_relu(last_relu, "new_conv_logits", NUM_FILTERS, dropout_ratio=self.dropout_ratio,
                                        is_training=self.is_train)
        classes = self.num_classes if self.num_classes > 2 else 1
        logits = unet._create_conv_relu(conv_logits, "new_logits", classes, self.dropout_ratio, is_training=self.is_train)

        loss_softmax = unet.loss(logits, label, weight, self.num_classes)
        loss_angle = unet.angle_loss(angle_pred, angle_label, weight)

        total_loss = loss_softmax + loss_angle #tf.add_n(losses, name='total_loss')
        return logits, total_loss, last_relu, angle_pred

    def build_model(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.acc_file = os.path.join(checkpoint_dir, "accuracy.csv")
        cpu, gpu = func.find_devices()
        tf_dev = gpu if gpu != "" else cpu

        with tf.Graph().as_default() as graph, tf.device(cpu):
            checkpoint = func.find_last_checkpoint(checkpoint_dir)
            print("Restoring checkpoint %i.." % checkpoint, flush=True)
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'model_%06d.ckpt.meta' % checkpoint))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))
            checkpoint += 1

            self.is_train = graph.get_tensor_by_name("Placeholder:0")
            self.placeholder_img = graph.get_tensor_by_name("images:0")
            self.placeholder_label = graph.get_tensor_by_name("labels:0")
            self.placeholder_weight = graph.get_tensor_by_name("weight:0")
            self.placeholder_angle_label = graph.get_tensor_by_name("angle_labels:0")
            self.placeholder_prior = graph.get_tensor_by_name("prior:0")

            # Load saved global_step.
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            # New adam optimizer.
            opt = tf.train.AdamOptimizer(learning_rate=train_detection.BASE_LR, name='MyNewAdam')

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                logits, loss, last_relu, angle_pred = self._loss(self.placeholder_img, self.placeholder_label,
                                                                 self.placeholder_weight, self.placeholder_angle_label,
                                                                 self.placeholder_prior, graph, scope)
                self.outputs = (logits, loss, last_relu, angle_pred)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                grads = opt.compute_gradients(loss)

            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            variable_averages = tf.train.ExponentialMovingAverage(func.MOVING_AVERAGE_DECAY, global_step, name='MyNewExponentialMovingAverage')
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            batchnorm_updates_op = tf.group(*update_ops)
            self.train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

            # Init added variables only, not variables loaded from checkpoint.
            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                self.sess.run(tf.variables_initializer(not_initialized_vars))

        return checkpoint


def run_finetuning(data_path=DET_DATA_DIR, checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2"),
                   output_checkpoint_dir=None,
                   train_prop=0.9, n_iters=10, with_augmentation=True, dropout_ratio=0,
                   learning_rate=train_detection.BASE_LR, set_random_seed=False,
                   num_classes=CLASSES, return_img=False):
    '''
    Run train and test iterations on unet2 for n_iters.

    Saves mettrics and checkpoints to checkpoint_dir.

    :param data_path: dir holding .npz files.
    :param checkpoint_dir: used to build latest checkpoint, also to store newly trained checkpoints only if output_checkpoint_dir is not None.
    :param output_checkpoint_dir: if not None used to store newly trained checkpoints.
    :param train_prop: proportion of each .npz file to be trained on, rest is reserved for test.
    :param n_iters: how many .npz files to iterate through.
    :param with_augmentation: whether to randomly flip horizontally and vertically (train data only).
    :param set_random_seed:
    :param return_img: whether to return segmentation and angle preds on test images
    :return:
      model_obj: TrainModel object.
      img: if return_img is true, return last iteration's predictions on test (list of tuples of segmentation & angle preds)
      iters: total number of iterations performed to train model_obj (picks up from last checkpoint)
    '''
    model_obj = FinetuneModel(data_path, train_prop, with_augmentation, dropout_ratio, learning_rate, set_random_seed, num_classes)
    start_iter = model_obj.build_model(checkpoint_dir)
    if output_checkpoint_dir:
        func.make_dir(output_checkpoint_dir)
        model_obj.checkpoint_dir = output_checkpoint_dir
    return train_detection.run_training_on_model(model_obj, start_iter, n_iters, return_img)