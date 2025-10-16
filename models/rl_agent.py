import tensorflow as tf
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, input_size=10, output_size=3):
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model
    
    def train(self, states, actions, rewards, next_states, done):
        """Train with experience replay including done flag"""
        # Store experience in memory
        for i in range(len(states)):
            self.memory.append((states[i], actions[i], rewards[i], next_states[i], done[i]))
        
        # Train when enough memories
        if len(self.memory) < 32:
            return
            
        # Random sample from memory
        minibatch = np.random.choice(len(self.memory), 32, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        done = np.array([self.memory[i][4] for i in minibatch])
        
        # Calculate target Q-values
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        
        for i in range(len(minibatch)):
            if done[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Train model
        self.model.fit(states, target, epochs=1, verbose=0)
        self._adjust_epsilon()
    
    def _adjust_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
