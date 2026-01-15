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
git clone [https://github.com/yourusername/mdp-visualization.git](https://github.com/yourusername/mdp-visualization.git)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Visit http://127.0.0.1:5000 in your web browser.

## Project Setup
├── app.py              # Flask server handling routes and API logic
├── mdp.py              # Core MDP logic (Transition models, Algorithms)
├── requirements.txt    # List of python dependencies
├── static/
│   └── style.css       # Styling for the Grid and UI
└── templates/
    └── index.html      # Frontend interface