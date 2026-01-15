from flask import Flask, render_template, jsonify, request
from mdp import GridWorldMDP

app = Flask(__name__)
mdp = GridWorldMDP(rows=6, cols=6)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_state', methods=['GET'])
def get_state():
    """
    Syncs the UI with the current backend state.
    Returns the grid values and calculated policy arrows as JSON.
    """
    return jsonify({'values': mdp.V.tolist(), 'policy': mdp.get_current_policy(), 'delta': 0})

@app.route('/step', methods=['POST'])
def step():
    """
    Main execution endpoint for the UI.
    Receives current algorithm and gamma from the frontend, executes one 
    iteration, and returns the updated state values and policy arrows.
    """
    data = request.json
    mdp.gamma, algo = float(data.get('gamma', 0.9)), data.get('algorithm', 'value')
    if algo == 'value':
        values, delta = mdp.value_iteration_step()
        policy = mdp.get_current_policy(is_value_iter=True)
    else:
        values, delta = mdp.policy_iteration_step()
        policy = mdp.get_current_policy(is_value_iter=False)
    return jsonify({'values': values, 'policy': policy, 'delta': delta})

@app.route('/clear_values', methods=['POST'])
def clear_values():
    """
    Resets learned values (V) and the current policy to zero/default.
    Preserves the existing map layout and obstacle positions.
    """
    mdp.reset_values()
    return jsonify({'status': 'success'})

@app.route('/reset_env', methods=['POST'])
def reset_env():
    """
    Triggers a full environment reset, regenerating random obstacle positions 
    while keeping the defined Goal and Trap terminal states.
    """
    mdp.reset_env()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)