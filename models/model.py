"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size


class Model(ABC):


    def __init__(self, seed, lr, optimizer=None, gpu_fraction=0.2):

        self.lr = lr
        self._optimizer = optimizer

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss = self.create_model()
            self.saver = tf.train.Saver()
        config=tf.ConfigProto(log_device_placement=False)
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(graph=self.graph, config=config)
        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def set_params(self, model_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, self.sess)
    
    def update_params(self, gradient_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            update_op = self.optimizer.apply_gradients(zip(gradient_params, all_vars))
            self.sess.run(update_op)
    
    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        # print("model params:")
        # # print(type(model_params), model_params[0])
        # for i in range(len(model_params)):
        #     print(model_params[i].shape)
        return model_params
    
    def get_gradients(self):
        with self.graph.as_default():
            # gradient_paras = tf.gradients(self.loss, tf.trainable_variables())
            # gradients = self.sess.run(gradient_paras,
            #                             feed_dict={
            #                                 self.features: self.last_features,
            #                                 self.labels: self.last_labels})
            gradients_paras = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
            gradients = self.sess.run(gradients_paras,
                                            feed_dict={
                                                self.features: self.last_features,
                                                self.labels: self.last_labels})
        # print("gradient params:")
        for i in range(len(gradients)):
            if i == 0 :
                gradients[i] = np.array(gradients[0])[1]
            else:
                gradients[i] = np.array(gradients[i])
        return gradients

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)

        models = self.get_params()
        gradients = self.get_gradients()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return comp, models, gradients

    def run_epoch(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size):
            
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            self.last_features = input_data
            self.last_labels = target_data
            with self.graph.as_default():
                self.sess.run(self.train_op,
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data
                    })

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, 'loss': loss}

    def close(self):
        self.sess.close()

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
