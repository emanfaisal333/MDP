import numpy as np
import random

class GridWorldMDP:
    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.prob_success = 0.8
        self.prob_random = 0.2  
        self.terminals = [(0, 3), (1, 3)] #(0, 3) is the Goal and (1, 3) is the Trap
        self.obstacles = []
        self.reset_env()

    def reset_env(self):
        self.obstacles = []
        num_obstacles = random.randint(0, 3)
        
        while len(self.obstacles) < num_obstacles:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            
            if (r, c) not in self.terminals and (r, c) != (0, 0) and (r, c) not in self.obstacles:
                self.obstacles.append((r, c))    

        self.reset_values()

    def reset_values(self):
        """Resets iteration data while keeping current map."""
        self.V = np.zeros((self.rows, self.cols))
        self.V[0, 3] = 10.0   # Goal Value
        self.V[1, 3] = -10.0  # Trap Value
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

    def _calculate_ev(self, r, c, action):
        """
        Calculates Expected Value (EV) for a given state-action pair 
        based on the transition logic:
        80% Intended Direction
        20% Uniform Random Direction (Noise)
        """
        # 1. Intended Outcome
        ns_intended = self.get_next_state(r, c, action)
        val_intended = self.V[ns_intended]

        # 2. Random Outcome (Average of all 4 possible moves)
        val_random_sum = 0
        for a in self.actions:
            ns_rand = self.get_next_state(r, c, a)
            val_random_sum += self.V[ns_rand]
        val_random = val_random_sum / 4.0

        # Weighted Sum
        return (self.prob_success * val_intended) + (self.prob_random * val_random)

    def value_iteration_step(self):
        new_V = np.copy(self.V)
        delta = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals or (r, c) in self.obstacles: continue
                
                q_vals = []
                for a in self.actions:
                    # R(s) + gamma * ExpectedValue(s')
                    ev = self._calculate_ev(r, c, a)
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
                    
                    old_v = self.V[r, c]
                    action = self.policy[r][c]
                    
                    # Update V using the fixed policy's action
                    ev = self._calculate_ev(r, c, action)
                    self.V[r, c] = self.get_reward(r, c) + self.gamma * ev
                    
                    delta_eval = max(delta_eval, abs(old_v - self.V[r, c]))
            if delta_eval < theta: break
            
        # Improvement
        stable = True
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals or (r, c) in self.obstacles: continue
                
                old_a = self.policy[r][c]
                best_a, max_q = 'UP', -float('inf')
                
                # Find argmax Q(s, a)
                for a in self.actions:
                    ev = self._calculate_ev(r, c, a)
                    q = self.get_reward(r, c) + self.gamma * ev
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
                    # Calculate best action based on current V (One-step lookahead)
                    best_a, max_q = 'UP', -float('inf')
                    for a in self.actions:
                        # Must use Expected Value for lookahead in stochastic env
                        ev = self._calculate_ev(r, c, a)
                        q = self.get_reward(r, c) + self.gamma * ev
                        
                        if q > max_q: max_q, best_a = q, a
                    row.append(best_a)
            derived.append(row)
        return derived