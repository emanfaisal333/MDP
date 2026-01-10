import numpy as np
import random

class GridWorldMDP:
    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma # 
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT'] # 
        
        # Transition Model Parameters 
        self.prob_success = 0.8  # 80% intended direction
        self.prob_fail = 0.2     # 20% random/stay in place
        
        # Terminal states [cite: 14, 15]
        self.terminals = [(0, 3), (1, 3)] 
        self.reset()

    def reset(self):
        self.V = np.zeros((self.rows, self.cols)) # Initialize values to 0
        # Default initial policy: all pointing 'UP'
        self.policy = [['UP' for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.terminals:
            self.policy[r][c] = 'TERM'

    def get_reward(self, r, c):
        if (r, c) == (0, 3): return 10.0   # Goal state [cite: 14]
        if (r, c) == (1, 3): return -10.0  # Terminal negative state [cite: 15]
        return -0.1                        # Step cost [cite: 16]

    def get_next_state(self, r, c, action):
        if action == 'UP': next_s = (max(r-1, 0), c)
        elif action == 'DOWN': next_s = (min(r+1, self.rows-1), c)
        elif action == 'LEFT': next_s = (r, max(c-1, 0))
        elif action == 'RIGHT': next_s = (r, min(c+1, self.cols-1))
        return next_s

    def value_iteration_step(self):
        """Task 2: Value Iteration logic [cite: 32]"""
        new_V = np.copy(self.V)
        delta = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals: continue
                
                q_values = []
                for action in self.actions:
                    ns = self.get_next_state(r, c, action)
                    # Stochastic Transitions using variables 
                    expected_v = (self.prob_success * self.V[ns]) + (self.prob_fail * self.V[r, c])
                    q_values.append(self.get_reward(r, c) + self.gamma * expected_v)
                
                best_q = max(q_values)
                delta = max(delta, abs(best_q - self.V[r, c]))
                new_V[r, c] = best_q
        
        self.V = new_V
        return self.V.tolist(), delta

    def policy_iteration_step(self):
        """Task 3: Policy Iteration logic [cite: 33]"""
        # Policy Evaluation
        theta = 0.001
        while True:
            delta_eval = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in self.terminals: continue
                    old_v = self.V[r, c]
                    action = self.policy[r][c]
                    ns = self.get_next_state(r, c, action)
                    # Use transition variables 
                    self.V[r, c] = self.get_reward(r, c) + self.gamma * (
                        self.prob_success * self.V[ns] + self.prob_fail * self.V[r, c]
                    )
                    delta_eval = max(delta_eval, abs(old_v - self.V[r, c]))
            if delta_eval < theta: break

        # Policy Improvement
        policy_stable = True
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals: continue
                old_action = self.policy[r][c]
                
                best_action = 'UP'
                max_q = -float('inf')
                for action in self.actions:
                    ns = self.get_next_state(r, c, action)
                    q = self.get_reward(r, c) + self.gamma * (
                        self.prob_success * self.V[ns] + self.prob_fail * self.V[r, c]
                    )
                    if q > max_q:
                        max_q = q
                        best_action = action
                
                self.policy[r][c] = best_action
                if old_action != best_action: policy_stable = False
        
        return self.V.tolist(), 0.0 if policy_stable else 1.0

    def get_current_policy(self, is_value_iter=True):
        if not is_value_iter:
            return self.policy
            
        derived_policy = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.terminals:
                    row.append('TERM')
                    continue
                if np.all(self.V == 0):
                    row.append('UP')
                    continue
                best_action = 'UP'
                max_q = -float('inf')
                for action in self.actions:
                    ns = self.get_next_state(r, c, action)
                    if self.V[ns] > max_q:
                        max_q = self.V[ns]
                        best_action = action
                row.append(best_action)
            derived_policy.append(row)
        return derived_policy