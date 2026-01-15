# 🤖 MDP Visualization: Value & Policy Iteration

A web-based interactive visualization tool for understanding **Markov Decision Processes (MDP)**. This project implements and visualizes **Value Iteration** and **Policy Iteration** algorithms on a stochastic Grid World environment using Python (Flask) and JavaScript.

## 🚀 Features

* **Interactive Grid World:** A 6x6 grid environment with:
    * 🟩 **Goal State:** Reward +10.00
    * 🟥 **Trap State:** Reward -10.00
    * ⬛ **Obstacles:** Randomly generated walls
* **Two Core Algorithms:** Switch instantly between:
    * **Value Iteration:** visualizes the gradual propagation of utility values.
    * **Policy Iteration:** visualizes the rapid stabilization of the optimal policy.
* **Dynamic Controls:**
    * **Step-by-Step Execution:** Watch the learning process one iteration at a time.
    * **Gamma Slider ($\gamma$):** Adjust the Discount Factor in real-time (0.0 to 1.0) to see how it affects the agent's foresight.
* **Real-Time Statistics:** Displays the current **Iteration Count** and **Convergence Delta ($\Delta$)**.
* **Stochastic Physics:** Simulates a "slippery" world where moves have an 80% success rate and a 20% chance of slipping sideways.

## 🛠️ Tech Stack

* **Backend:** Python, Flask, NumPy
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 📦 Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/emanfaisal333/MDP.git
cd mdp-visualization

```

### 2. Set up a Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
python app.py

```

### 5. Open in Browser

Visit `http://127.0.0.1:5000` in your web browser.

## 🎮 How to Use

1. **Select Algorithm:** Choose between "Value Iteration" or "Policy Iteration" from the dropdown.
2. **Set Gamma:** Use the slider to set the Discount Factor.
* *Low (e.g., 0.1):* Agent is short-sighted.
* *High (e.g., 0.9):* Agent plans for the long term.


3. **Step Iteration:** Click the button to run one loop of the algorithm. Watch the values update and arrows change.
4. **Check Convergence:** Continue stepping until the "Optimal Policy Reached!" message appears.
5. **Reset:**
* **Reset Values:** Clears the calculated numbers but keeps the walls/obstacles.
* **New Map:** Generates a completely new random arrangement of obstacles.



## 📂 Project Structure

```plaintext
├── app.py              # Flask server handling routes and API logic
├── mdp.py              # Core MDP logic (Transition models, Algorithms)
├── requirements.txt    # List of python dependencies
├── static/
│   └── style.css       # Styling for the Grid and UI
└── templates/
    └── index.html      # Frontend interface

```

## 🧠 Algorithms Explained

### Value Iteration

Calculates the utility of every state by iteratively applying the **Bellman Update Equation**. It propagates rewards from the goal state outwards until the values stabilize (converge).


### Policy Iteration

Alternates between two phases:

1. **Policy Evaluation:** Calculates the value of the current policy.
2. **Policy Improvement:** Updates the policy to be greedy with respect to the new values.
*This method typically converges in fewer iterations than Value Iteration.*

```