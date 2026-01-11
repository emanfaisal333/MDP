import numpy as np
import random

class GridWorldMDP:
    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.prob_success = 0.8
        self.prob_fail = 0.2
        self.terminals = [(0, 3), (1, 3)]
        self.obstacles = []
        self.reset_env()

    def reset_env(self):
        """Generates new obstacles in Row 1 and Row 3."""
        self.obstacles = []
        for r in [1, 3]:
            valid_cols = [c for c in range(self.cols) if (r, c) not in self.terminals]
            if valid_cols:
                self.obstacles.append((r, random.choice(valid_cols)))
        self.reset_values()

    def reset_values(self):
        """Resets iteration data while keeping current map."""
        self.V = np.zeros((self.rows, self.cols))
        self.policy = [['UP' for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.terminals: self.policy[r][c] = 'TERM'
        for r, c in self.obstacles: self.policy[r][c] = 'OBS'

    def get_reward(self, r, c):
        if (r, c) == (0, 3): return 10.0
        if (r, c) == (1, 3): return -10.0
        return -0.1

    def get_next_state(self, r, c, action):
        if action == 'UP': ns = (max(r-1, 0), c)
        elif action == 'DOWN': ns = (min(r+1, self.rows-1), c)
        elif action == 'LEFT': ns = (r, max(c-1, 0))
        elif action == 'RIGHT': ns = (r, min(c+1, self.cols-1))
        return (r, c) if ns in self.obstacles else ns

    def value_iteration_step(self):
        new_V = np.copy(self.V)
        delta = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals or (r, c) in self.obstacles: continue
                q_vals = []
                for a in self.actions:
                    ns = self.get_next_state(r, c, a)
                    ev = (self.prob_success * self.V[ns]) + (self.prob_fail * self.V[r, c])
                    q_vals.append(self.get_reward(r, c) + self.gamma * ev)
                best_q = max(q_vals)
                delta = max(delta, abs(best_q - self.V[r, c]))
                new_V[r, c] = best_q
        self.V = new_V
        return self.V.tolist(), delta

    def policy_iteration_step(self):
        # Evaluation
        theta = 0.001
        while True:
            delta_eval = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in self.terminals or (r, c) in self.obstacles: continue
                    old_v, action = self.V[r, c], self.policy[r][c]
                    ns = self.get_next_state(r, c, action)
                    self.V[r, c] = self.get_reward(r, c) + self.gamma * (self.prob_success * self.V[ns] + self.prob_fail * self.V[r, c])
                    delta_eval = max(delta_eval, abs(old_v - self.V[r, c]))
            if delta_eval < theta: break
        # Improvement
        stable = True
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals or (r, c) in self.obstacles: continue
                old_a, best_a, max_q = self.policy[r][c], 'UP', -float('inf')
                for a in self.actions:
                    ns = self.get_next_state(r, c, a)
                    q = self.get_reward(r, c) + self.gamma * (self.prob_success * self.V[ns] + self.prob_fail * self.V[r, c])
                    if q > max_q: max_q, best_a = q, a
                self.policy[r][c] = best_a
                if old_a != best_a: stable = False
        return self.V.tolist(), 0.0 if stable else 1.0

    def get_current_policy(self, is_value_iter=True):
        if not is_value_iter: return self.policy
        derived = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.terminals: row.append('TERM')
                elif (r, c) in self.obstacles: row.append('OBS')
                elif np.all(self.V == 0): row.append('UP')
                else:
                    best_a, max_q = 'UP', -float('inf')
                    for a in self.actions:
                        ns = self.get_next_state(r, c, a)
                        if self.V[ns] > max_q: max_q, best_a = self.V[ns], a
                    row.append(best_a)
            derived.append(row)
        return derived
