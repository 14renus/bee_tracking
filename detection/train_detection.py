import os
import re
import time, math
from random import shuffle, randint, seed
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from . import unet
from utils import func
from utils.paths import DET_DATA_DIR, CHECKPOINT_DIR
from utils.func import DS, GPU_NAME, NUM_LAYERS, NUM_FILTERS, CLASSES, clipped_sigmoid
from plots import segm_map

BATCH_SIZE = 4
BASE_LR = 0.0001
SEED = 0

tf.logging.set_verbosity(tf.logging.ERROR)


def flip_h(data):
    data = np.flip(data, axis=2)

    for fr in range(data.shape[0]):
        ang = data[fr,2,:,:]
        is_1 = data[fr,1,:,:] == 1
        ang[is_1] = ((math.pi - ang[is_1]*2*math.pi) % (2 * math.pi)) / (2*math.pi)
        data[fr,2,:,:] = ang
    return data


def flip_v(data):
    data = np.flip(data, axis=3)

    for fr in range(data.shape[0]):
        ang = data[fr,2,:,:]
        is_1 = data[fr,1,:,:] == 1
        ang[is_1] = 1 - ang[is_1]
        data[fr,2,:,:] = ang
    return data


class TrainModel:

    def __init__(self, data_path, train_prop, with_augmentation, dropout_ratio=0, learning_rate=BASE_LR, loss_upweight=10, set_random_seed=False, num_classes=3):
        self.data_path = data_path
        self.input_files = [f for f in os.listdir(data_path) if re.search('npz', f)]
        self.set_random_seed = set_random_seed
        if self.set_random_seed:
          seed(SEED)
          np.random.seed(SEED)
        else:
          shuffle(self.input_files)
        self.train_prop = train_prop
        self.with_augmentation = with_augmentation
        self.dropout_ratio = dropout_ratio
        self.learning_rate = learning_rate
        self.loss_upweight = loss_upweight
        self.num_classes = num_classes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()
        None

    def __del__(self):
        self.sess.close()


    def _loss(self, img, label, weight, angle_label, prior):
        logits, last_relu, angle_pred = unet.create_unet2(NUM_LAYERS, NUM_FILTERS, img, self.is_train, prev=prior,
                                                          dropout_ratio=self.dropout_ratio,
                                                          set_random_seed=self.set_random_seed, num_classes=self.num_classes)
        loss_softmax = unet.loss(logits, label, weight, self.num_classes)
        loss_angle = unet.angle_loss(angle_pred, angle_label, weight)

        total_loss = loss_softmax + loss_angle #tf.add_n(losses, name='total_loss')
        return logits, total_loss, last_relu, angle_pred, loss_softmax, loss_angle

    def build_model(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        cpu, gpu = func.find_devices()
        tf_dev = gpu if gpu != "" else cpu

        with tf.Graph().as_default(), tf.device(cpu):
            if self.set_random_seed:
                tf.set_random_seed(SEED)

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.is_train = tf.placeholder(tf.bool, shape=[])
            self.placeholder_img = tf.placeholder(tf.float32, shape=(None, DS, DS, 1), name="images")
            self.placeholder_label = tf.placeholder(tf.uint8, shape=(None, DS, DS), name="labels")
            self.placeholder_weight = tf.placeholder(tf.float32, shape=(None, DS, DS), name="weight")
            self.placeholder_angle_label = tf.placeholder(tf.float32, shape=(None, DS, DS), name="angle_labels")
            self.placeholder_prior = tf.placeholder(tf.float32, shape=(None, DS, DS, NUM_FILTERS), name="prior")

            with tf.device(tf_dev), tf.name_scope('%s_%d' % (GPU_NAME, 0)) as scope:
                logits, total_loss, last_relu, angle_pred,loss_softmax, loss_angle = self._loss(self.placeholder_img, self.placeholder_label,
                                                                 self.placeholder_weight, self.placeholder_angle_label,
                                                                 self.placeholder_prior)
                self.outputs = (logits, total_loss, last_relu, angle_pred, loss_softmax, loss_angle)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                grads = opt.compute_gradients(total_loss)

            #grads = self._average_gradients(grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
            variable_averages = tf.train.ExponentialMovingAverage(func.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            batchnorm_updates_op = tf.group(*update_ops)
            self.train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
            self.saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            self.checkpoint_dir = checkpoint_dir
            self.acc_file = os.path.join(checkpoint_dir, "accuracy.csv")
            checkpoint = func.find_last_checkpoint(checkpoint_dir)
            if checkpoint > 0:
                print("Restoring checkpoint %i.." % checkpoint, flush=True)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model_%06d.ckpt' % checkpoint))
                checkpoint += 1
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                checkpoint = 0
            init = tf.local_variables_initializer()
            self.sess.run(init)

        return checkpoint


    def _accuracy(self, step, loss, logits, angle_preds, batch_data, loss_softmax, loss_angle):
        '''
        Calculate metrics for train or test step.

        :param step: Frame step in batch_data to use as labels.
        :param loss: cross entropy + regression angle loss already calculated from tf model.
        :param logits: Raw class preds before softmax/sigmoid [BATCH_SIZE, DS, DS, num_classes]
        :param angle_preds: Regression preds for angle
        :param batch_data: Full data [BATCH_SIZE, nb_frames, 4, DS, DS]
        :return: Tuple of metrics
                 - Boolean to indicate train (0) or test (1)
                 - loss: passed from model
                 - bg: "background overlap" = (correct class = 0 and angle < 0) / # background pixels
                 - fg: "foreground overlap" = (correct class !=0) / # foreground pixels
                 Just for foreground pixels:
                 - fg_error: "class error" = incorrect class / # foreground pixels
                 - angle_error: "angle error" = mean difference in angle
        '''
        batch_data = batch_data[:, step, :, :, :]

        if self.num_classes > 2:
            pred_class = np.argmax(logits, axis=3)
        else:
            pred_class = np.round(clipped_sigmoid(logits.squeeze()))
        pred_angle = angle_preds[:, :, :, 0]

        lb = batch_data[:,1,:,:]
        angle = batch_data[:,2,:,:]
        is_bg = (lb == 0)
        is_fg = np.logical_not(is_bg)
        n_fg = np.sum(is_fg)
        # Background accuracy. Correct if pred class 0 and angle < 0.
        # bg = float(np.sum((pred_class[is_bg] == 0) & (pred_angle[is_bg] < 0)))/np.sum(is_bg)
        bg = float(np.sum(pred_class[is_bg] == 0))/np.sum(is_bg)
        fg = 0
        fg_err = np.max(lb)
        angle_err = 0
        if n_fg > 0:
            # Foreground accuracy. Correct if pred class != 0.
            fg = float(np.sum(pred_class[is_fg] != 0))/n_fg
            # Foreground error. Incorrect if pred class != label class.
            fg_err = np.mean(lb[is_fg] != pred_class[is_fg])
            # Foreground angle error. Abs difference in angle pred and label
            angle_err = np.mean(np.abs(pred_angle[is_fg] - angle[is_fg]))
        return np.array([0, loss, bg, fg, fg_err, angle_err, loss_softmax, loss_angle])

    def _sample_offsets(self, data):
        '''
        To generate a batch of size n, randomly choose n DS x DS patches. Also randomly flip horizontally and vertically.

        :param data: data from 1 npz file (1 sequence of frames). shape = (num_frames, 4, H, W)
        '''
        res = np.zeros((BATCH_SIZE, data.shape[0], data.shape[1], DS, DS))
        for i in range(BATCH_SIZE):
            # Randomly sample x and y offset. Randomly decide to flip horizontally and vertically if with_augmentation.
            off_x, off_y, fh, fv = randint(0, data.shape[2]-DS), randint(0, data.shape[3]-DS), randint(0, 1), randint(0, 1)
            if not self.with_augmentation:
                fh, fv = 0, 0
            cut_data = np.copy(data[:,:,off_x:(off_x+DS),off_y:(off_y+DS)])
            if fh:
                cut_data = flip_h(cut_data)
            if fv:
                cut_data = flip_v(cut_data)
            res[i] = cut_data
        return res, np.zeros((BATCH_SIZE, DS, DS, NUM_FILTERS), dtype=np.float32)


    def _input_batch(self, step, batch_data, last_relus, is_train):
        return {self.placeholder_img: np.resize(batch_data[:,step,0,:,:],(BATCH_SIZE, DS, DS, 1)),
                self.placeholder_label: batch_data[:,step,1,:,:], self.placeholder_angle_label: batch_data[:,step,2,:,:],
                self.placeholder_weight: batch_data[:,step,3,:,:]*(self.loss_upweight-1)+1,
                self.placeholder_prior: last_relus, self.is_train: is_train}


    def run_test(self, batch_data, start_step, last_relus, return_img):
        t1 = time.time()

        res_img = []
        accuracy_t = np.zeros((8))
        for step in range(start_step, batch_data.shape[1]):
            outs = self.sess.run(self.outputs, feed_dict=self._input_batch(step, batch_data, last_relus, False))
            last_relus = outs[2]
            accuracy_t += self._accuracy(step, outs[1], outs[0], outs[3], batch_data, outs[4], outs[5])
            if (step == (batch_data.shape[1]-1)) and return_img:
                for i in range(BATCH_SIZE):
                    im_segm = segm_map.plot_segm_map_np(batch_data[i, step, 0, :, :], np.argmax(outs[0][i], axis=2))
                    im_angle = segm_map.plot_angle_map_np(batch_data[i,step,0,:,:], outs[3][i])
                    res_img.append((im_segm, im_angle))

        accuracy_t = accuracy_t / (batch_data.shape[1] - start_step)
        accuracy_t[0] = 1
        print("TEST - time: %.3f min, loss: %.3f, class loss: %.3f, angle loss: %.3f, background overlap: %.3f, foreground overlap: %.3f, class error: %.3f, angle error: %.3f" % ((time.time() - t1) / 60, accuracy_t[1], accuracy_t[6],accuracy_t[7],accuracy_t[2], accuracy_t[3], accuracy_t[4], accuracy_t[5]), flush=True)
        with open(self.acc_file, 'a') as f:
            np.savetxt(f, np.reshape(accuracy_t, (1,-1)), fmt='%.5f', delimiter=',', newline='\n')
        return res_img


    def run_train_test_iter(self, itr, return_img):
        '''
        Run training and test on 1 .npz file.

        :param itr: index of .npz file to load
        :param return_img: whether to return test images
        :return:
        '''
        file = self.input_files[itr % len(self.input_files)]
        npz = np.load(os.path.join(self.data_path, file))
        data = npz['data']
        t1 = time.time()
        train_steps = int(data.shape[0]*self.train_prop) # Number of sequential frames to train on.
        batch_data, last_relus = self._sample_offsets(data)

        accuracy_t = np.zeros((8))
        for step in range(train_steps):
            _, outs = self.sess.run([self.train_op, self.outputs],
                                    feed_dict=self._input_batch(step, batch_data, last_relus, True))
            last_relus = outs[2]
            accuracy_t += self._accuracy(step, outs[1], outs[0], outs[3], batch_data, outs[4], outs[5])

        accuracy_t = accuracy_t / train_steps
        accuracy_t[0] = 0
        print("TRAIN - time: %.3f min, loss: %.3f, class loss: %.3f, angle loss: %.3f, background overlap: %.3f, foreground overlap: %.3f, class error: %.3f, angle error: %.3f" % ((time.time() - t1) / 60, accuracy_t[1], accuracy_t[6],accuracy_t[7], accuracy_t[2], accuracy_t[3], accuracy_t[4], accuracy_t[5]), flush=True)
        with open(self.acc_file, 'a') as f:
            np.savetxt(f, np.reshape(accuracy_t, (1,-1)), fmt='%.5f', delimiter=',', newline='\n')

        img = []
        if train_steps < batch_data.shape[1]:
            img = self.run_test(batch_data, train_steps, last_relus, return_img)
        return img

def run_training_on_model(model_obj, start_iter, n_iters, return_img):
    for i in range(start_iter, start_iter + n_iters):
        print("ITERATION: %i" % i, flush=True)
        img = model_obj.run_train_test_iter(i, return_img=return_img)
        model_obj.saver.save(model_obj.sess, os.path.join(model_obj.checkpoint_dir, 'model_%06d.ckpt' % i))
    return model_obj, img, start_iter + n_iters


def run_training(data_path=DET_DATA_DIR, checkpoint_dir=os.path.join(CHECKPOINT_DIR, "unet2"),
                 output_checkpoint_dir=None,
                 train_prop=0.9, n_iters=10, with_augmentation=True, dropout_ratio=0, learning_rate=BASE_LR,
                 loss_upweight=10,
                 set_random_seed=False,
                 num_classes=CLASSES, return_img=False):
    '''
    Run train and test iterations on unet2 for n_iters.

    Saves mettrics and checkpoints to checkpoint_dir.

    :param data_path: dir holding .npz files.
    :param checkpoint_dir: used to build latest checkpoint and store newly trained checkpoints.
    :param output_checkpoint_dir: if not None used to store newly trained checkpoints.
    :param train_prop: proportion of each .npz file to be trained on, rest is reserved for test.
    :param n_iters: how many .npz files to iterate through.
    :param with_augmentation: whether to randomly flip horizontally and vertically (train data only).
    :param loss_upweight: positive weight to upweight pixels when calculating average loss.
                          assumes weight placeholder is 0 where pixels should not be upweighted.
    :param set_random_seed:
    :param return_img: whether to return segmentation and angle preds on test images
    :return:
      model_obj: TrainModel object.
      img: if return_img is true, return last iteration's predictions on test (list of tuples of segmentation & angle preds)
      iters: total number of iterations performed to train model_obj (picks up from last checkpoint)
    '''
    model_obj = TrainModel(data_path, train_prop, with_augmentation, dropout_ratio, learning_rate, loss_upweight, set_random_seed, num_classes)
    start_iter = model_obj.build_model(checkpoint_dir)
    if output_checkpoint_dir:
        func.make_dir(output_checkpoint_dir)
        model_obj.checkpoint_dir = output_checkpoint_dir
    return run_training_on_model(model_obj, start_iter, n_iters, return_img)


