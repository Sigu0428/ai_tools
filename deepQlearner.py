import collections
import numpy as np
import keras
from keras import layers
import random

class deepQLearner():
    input_size = 6

    def __init__(self, hyper_parameters, log):
        self.log = log
        self.hp = hyper_parameters
        self.target_network = self.initNeuralNetwork()
        self.behavior_network = self.initNeuralNetwork(self.target_network)

        self.network_copy_counter = 0
        self.train_counter = 0
        self.t = 0
        self.G = 0
        self.S_prev = None
        self.A_prev = None
        self.prior_exp = collections.deque(maxlen=self.hp['prior_exp_window'])

    def initNeuralNetwork(self, copy_from=None):
        if copy_from is None:
            inputs = keras.Input(shape=(self.input_size,), name="state")
            x = layers.Dense(self.hp['NN_structure'][0], activation=self.hp['activation'], name="dense_0")(inputs)
            for i, layer in enumerate(self.hp['NN_structure'][1:]):
                x = layers.Dense(layer, activation=self.hp['activation'], name="dense_" + str(i+1))(x)
            outputs = layers.Dense(1, activation=keras.activations.linear, name="expected_return")(x)
            Q_network = keras.Model(inputs=inputs, outputs=outputs)
            Q_network.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.hp['learning_rate']),
                loss=keras.losses.Huber(),
                metrics=[keras.metrics.MeanSquaredError()],
            )
            return Q_network

        model_copy= keras.models.clone_model(copy_from)
        model_copy.build((None, self.input_size))
        model_copy.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.hp['learning_rate']),
                loss=keras.losses.Huber(),
                metrics=[keras.metrics.MeanSquaredError()],
            )
        model_copy.set_weights(copy_from.get_weights()) 
        return model_copy

    def getAction(self, state_action_returns, A_possible, greedy = False):
        if len(A_possible) == 0:
            return -1
        if (random.random() < self.hp['explore_chance']) and not greedy:
            return random.choice(A_possible)
        return A_possible[np.argmax(state_action_returns)]
    
    def predictStateActionReturns(self, S, A_possible):
        if len(A_possible) > 0:
            return self.behavior_network(np.vstack((np.repeat(S[:, None], len(A_possible), axis=1), np.array(A_possible))).T)
        else:
            return None
        
    def OnGameEnd(self):
        self.log.appendData('returns', self.G)
        self.G = 0
        self.t = 0
        self.A_prev = None
        self.S_prev = None

    def updateOnExp(self):
        batch_size = min(len(self.prior_exp), self.hp['batch_size'])
        random_idxs = np.random.choice(np.arange(len(self.prior_exp)), replace=False, size=(min(batch_size, batch_size*self.hp['epochs'])))
        SAs = np.zeros((batch_size, self.input_size))
        Gs = np.zeros((batch_size, 1))
        for batch_idx, exp_idx in enumerate(random_idxs):
            (A, S, S_PRIME, R, A_POSSIBLE) = self.prior_exp[exp_idx]
            #print("function inputs", (A, S, S_PRIME, R, A_POSSIBLE))
            #print("network output", self.target_network(np.vstack((np.repeat(S_PRIME[:, None], len(A_POSSIBLE), axis=1), np.array(A_POSSIBLE))).T))
            #print("network input", np.vstack((np.repeat(S_PRIME[:, None], len(A_POSSIBLE), axis=1), np.array(A_POSSIBLE))).T)
            max_Q = np.max(self.target_network(np.vstack((np.repeat(S_PRIME[:, None], len(A_POSSIBLE), axis=1), np.array(A_POSSIBLE))).T))
            SAs[batch_idx, :] = np.vstack((S[:, None], np.array((A, )))).T
            Q_prev = self.target_network(np.vstack((S[:, None], np.array([A]))).T)
            if self.hp['bootstrapping']:
                Gs[batch_idx, :] = Q_prev + self.hp["alpha"]*(R + self.hp['gamma']*max_Q - Q_prev)
            else:
                Gs[batch_idx, :] = R
            #print("state", S, "action", A, "reward:", R, "update target", R + self.hp['discount']*max_Q)

        if True:
            history = self.behavior_network.fit(
                SAs,
                Gs,
                batch_size=batch_size,
                epochs=self.hp['epochs'],
                verbose=0
            )
            loss = np.mean(history.history["loss"])
            self.log.appendData('loss', loss)
        else:
            history = self.behavior_network.train_on_batch(
                SAs,
                Gs,
                return_dict = True
            )
            self.loss_hist.append(np.mean(history["loss"]))

    def OnEndOfTurn(self, S_PRIME, A_PRIME, R, A_POSSIBLE):
        if self.A_prev is not None and len(A_POSSIBLE) > 0:
            self.prior_exp.append((self.A_prev, self.S_prev, S_PRIME, R, A_POSSIBLE))
        self.A_prev = A_PRIME
        self.S_prev = S_PRIME
        
        self.G = self.G + (self.hp['gamma']**self.t)*R
        
        self.train_counter += 1
        if self.train_counter > self.hp['training_frequency']:
            self.train_counter = 0
            self.updateOnExp()
        
        self.network_copy_counter += 1
        if self.network_copy_counter > self.hp['target_update_frequency']:
            self.network_copy_counter = 0
            self.target_network = self.initNeuralNetwork(self.behavior_network)