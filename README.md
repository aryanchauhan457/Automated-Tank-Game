# 🎯 Automated Tank Game (Reinforcement Learning)

A self-learning tank shooter game built using **Python**, **Pygame**, and **Q-Learning** (using NumPy). The AI tank learns to rotate and fire at targets, with persistent training stored in a Q-table across sessions.

---

## 🧠 Features

- 🤖 Autonomous tank agent using **Q-Learning**
- 🎮 Built with **Pygame** for real-time interaction and graphics
- 💾 **Persistent Q-table** (`q_table.npy`) to retain learning across runs
- 📈 Reward-based learning:
  - Hits give positive reward
  - Misses give penalty
  - Time steps incur small penalties (to encourage quicker action)
- 🧠 Discrete state/action space for efficient tabular Q-learning

---

## 📁 Project Structure

📂 project-root/
├── automated_tank_game.py # Main RL game script
├── tank.png # Tank turret image
├── bullet.png # Bullet image
├── target.png # Target image
├── icon.png # Window icon
├── q_table.npy # Saved Q-table (auto-created)
└── README.md # This file

---

## 📦 Requirements

Install the required dependencies:

```bash
pip install pygame numpy

🚀 How It Works
State Space:

Discretized tank turret angle (36 bins: 0–360°)

Discretized distance to target (10 bins)

Action Space:

0: Rotate Left

1: Rotate Right

2: Fire

Reward System:

+100 → Successful hit

-10 → Bullet missed (off screen)

-1 → Time step penalty

Epsilon Decay:

Starts with high exploration (ε = 1.0)

Gradually shifts to exploitation (ε → 0.05)

🕹️ How to Run
```bash
python automated_tank_game.py

💡 The AI controls the tank — no user input needed.

The tank rotates and fires bullets toward the target.

After every game episode, it updates the Q-table based on rewards.

The Q-table is automatically saved as q_table.npy after training.

💾 Persistent Learning
On each run:

If q_table.npy exists → it loads and continues training.

If not → it initializes a new Q-table.

After training, it automatically saves the updated Q-table to the same file.

You can reset the learning by deleting q_table.npy.

🛠️ Possible Extensions
Add moving targets or random speeds

Save/load game progress visually (e.g., using plots)

Upgrade to Deep Q-Learning (DQN) with TensorFlow or PyTorch

Create a training vs testing mode toggle

🙋 Author
Aryan
B.Tech CSE Student | Rajput ⚔️
Interested in intelligent agents, AI games, and reinforcement learning.
