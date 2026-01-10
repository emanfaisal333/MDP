from flask import Flask, render_template, jsonify, request
from mdp import GridWorldMDP

app = Flask(__name__)
mdp = GridWorldMDP()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_state', methods=['GET'])
def get_state():
    # Show initial 0.0 values and 'UP' default arrows [cite: 23, 24]
    return jsonify({
        'values': mdp.V.tolist(),
        'policy': mdp.get_current_policy(),
        'delta': 0
    })

@app.route('/step', methods=['POST'])
def step():
    data = request.json
    mdp.gamma = float(data.get('gamma', 0.9)) # User-controlled [cite: 17]
    algo = data.get('algorithm', 'value')
    
    if algo == 'value':
        values, delta = mdp.value_iteration_step()
        policy = mdp.get_current_policy(is_value_iter=True)
    else:
        values, delta = mdp.policy_iteration_step()
        policy = mdp.get_current_policy(is_value_iter=False)
        
    return jsonify({'values': values, 'policy': policy, 'delta': delta})

@app.route('/reset', methods=['POST'])
def reset():
    mdp.reset()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)