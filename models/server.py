import numpy as np
import timeout_decorator
from gradientCompression import SignSGD
import traceback
from utils.logger import Logger

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

L = Logger()
logger = L.get_logger()

class Server:
    def __init__(self, client_model, clients=[]):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.gradients = None
        self.selected_clients = []
        self.all_clients = clients
        self.updates = []

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, deadline=-1):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            deadline: -1 for unlimited; >0 for each client's deadline
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in self.all_clients:
            c.model.set_params(self.model)
        # for c in self.all_clients:
        #     if self.gradients is not None:
        #         c.model.update_params(self.gradients)
        simulate_time = 0
        for c in clients:
            # c.model.set_params(self.model)
            try:
                # set deadline 
                c.set_deadline(deadline)
                # training
                logger.debug('client {} starts training...'.format(c.id))
                simulate_time_c, comp, num_samples, models, gradients = c.train(num_epochs, batch_size, minibatch)
                dict_value = c.test()
                logger.info('client {} simulate_time: {} accuracy: {} loss: {}'.format(c.id, simulate_time_c, dict_value['accuracy'], dict_value['loss']))
                # !!!!!
                compressed_update_gradients, before_compressed_bits, after_compressed_bits = SignSGD.GradientCompress(gradients)
                if simulate_time_c > simulate_time:
                    simulate_time = simulate_time_c
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                # uploading 
                # self.updates.append((c.id, num_samples, models))
                self.updates.append((c.id, num_samples, compressed_update_gradients))
                logger.info('client {} upload successfully!'.format(c.id))
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                logger.info('client {} failed: timeout!'.format(c.id))
                simulate_time = deadline
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                traceback.print_exc()
        logger.info('simulation time: {}'.format(simulate_time))
        return sys_metrics

    def update_model(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            used_client_ids = [cid for (cid, client_samples, client_model) in self.updates]
            total_weight = 0.
            base = [0] * len(self.updates[0][2])
            for (cid, client_samples, client_model) in self.updates:
                total_weight += client_samples
                for i, v in enumerate(client_model):
                    base[i] += (client_samples * v.astype(np.float32))
            for c in self.all_clients:
                if c.id not in used_client_ids:
                    # c was not trained in this round
                    params = c.model.get_params()
                    total_weight += c.num_train_samples  # assume that all train_data is used to update
                    for i, v in enumerate(params):
                        base[i] += (c.num_train_samples * v.astype(np.float32))
            averaged_soln = [v / total_weight for v in base]
            self.model = averaged_soln
        else:
            logger.info('round failed, global model maintained.')
        self.updates = []

    def update_gradient(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            used_client_ids = [cid for (cid, client_samples, client_gradient) in self.updates]
            total_weight = 0.
            base = [0] * len(self.updates[0][2])
            for (cid, client_samples, client_gradient) in self.updates:
                total_weight += client_samples
                for i, v in enumerate(client_gradient):
                    base[i] += (client_samples * v.astype(np.float32))
            for c in self.all_clients:
                if c.id not in used_client_ids:
                    # c was not trained in this round
                    total_weight += c.num_train_samples  # assume that all train_data is used to update

            averaged_gradient = [grad / total_weight for grad in base]
            self.gradients = averaged_gradient
            self.model = self.client_model.update_params(self.gradients)
        else:
            logger.info('round failed, global model maintained.')
        self.updates = []
        
    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.all_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()