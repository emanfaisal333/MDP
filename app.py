from flask import Flask, render_template, jsonify, request
from mdp import GridWorldMDP

app = Flask(__name__)
mdp = GridWorldMDP()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_state', methods=['GET'])
def get_state():
    return jsonify({'values': mdp.V.tolist(), 'policy': mdp.get_current_policy(), 'delta': 0})

@app.route('/step', methods=['POST'])
def step():
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
    mdp.reset_values()
    return jsonify({'status': 'success'})

@app.route('/reset_env', methods=['POST'])
def reset_env():
    mdp.reset_env()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)