# 🏎️ Box Box Bot

> *"Box box, the agent has forgotten how to turn."*

Box Box Bot is an F1-inspired reinforcement learning project that trains AI drivers with distinct personalities to race around a track. Using PPO (Proximal Policy Optimisation) and a custom reward-shaping wrapper around OpenAI Gymnasium's CarRacing environment, four drivers each learn to drive from pure pixel input — no GPS, no lap times, just a 96×96 image and a lot of trial and error.

The project tracks each driver's development from rookie chaos to something vaguely resembling racecraft, comparing their lap performance, steering smoothness, and consistency — because every AI deserves a pit wall.

---

## 🏁 The Driver Lineup

| # | Driver | Vibe | Speed Bonus | Off-Track Penalty | Smooth Bonus |
|---|--------|------|-------------|-------------------|--------------|
| 💥 #33 | **Divebomber** | Fast and reckless. Asks questions later. | 0.5 | -0.3 | 0.0 |
| 🧘 #44 | **Tyre Whisperer** | Smooth and consistent. Tyres are a lifestyle. | 0.1 | -1.5 | 0.4 |
| 🏆 #1 | **Sunday Specialist** | Balanced. Reliable. Quietly dangerous. | 0.2 | -0.8 | 0.2 |
| 🎸 #55 | **Smooth Operator** | Calculated, smooth, always in the mix. Never panics. | 0.15 | -1.2 | 0.5 |

Each driver is shaped by tweaking three reward levers:
- **Speed bonus** — amplifies positive (on-track) reward. Higher = more aggressive pace.
- **Off-track penalty** — extra punishment for leaving the tarmac. Higher = hates the gravel.
- **Smooth bonus** — rewards gentle, consistent inputs. Higher = silkier racecraft.

---

## 🛠️ How It Works

The environment is `CarRacing-v3` from OpenAI Gymnasium. The agent receives a 96×96 RGB image at each step and outputs three continuous actions: **steering**, **throttle**, and **brake**.

Training uses **PPO (Proximal Policy Optimisation)** via Stable Baselines3 with a CNN policy — the agent learns directly from pixels. Each driver is trained for 1 million steps across 4 parallel environments, with checkpoints saved at 100k, 300k, 600k, and 1M steps so you can watch the rookie-to-champion arc.

---

## 📁 Project Structure

```
box-box-bot/
├── agents/
│   ├── personalities.py  # F1DriverEnv wrapper — reward shaping per driver
│   ├── train.py          # full championship training script
│   └── evaluate.py       # record GIFs and collect telemetry stats
├── notebooks/
│   └── explore.ipynb     # pre-season shakedown and results visualisation
├── results/              # CSVs land here after training
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/box-box-bot.git
cd box-box-bot
```

### 2. Create a conda environment

```bash
conda create -n boxboxbot python=3.10
conda activate boxboxbot
pip install -r requirements.txt
```

### 3. Verify the pipeline

```bash
python agents/train.py --test
```

This runs a quick 5k-step test to confirm everything is installed and working. Takes about a minute.

### 4. Full championship training

```bash
python agents/train.py
```

Trains all four drivers at checkpoints of 100k, 300k, 600k, and 1M steps. Saves models to `models/` and results to `results/`.

> ⏱️ Estimated time: 3–4 hours on a Mac (M-series chips are faster).
> 
> To train a single driver: `python agents/train.py --driver divebomber`

### 5. Record replays and standings

```bash
python agents/evaluate.py
```

Loads each trained model, runs a deterministic episode, saves a GIF to `replays/`, and prints the final championship standings.

### 6. Explore results

Open `notebooks/explore.ipynb` in VS Code or JupyterLab. Cells 10 and 11 load the evaluation summary and plot driver comparisons.

---

## 📊 Results

*Training in progress — results to follow.*

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| [Gymnasium](https://gymnasium.farama.org/) | CarRacing-v3 environment |
| [Stable Baselines3](https://stable-baselines3.readthedocs.io/) | PPO implementation |
| [imageio](https://imageio.readthedocs.io/) | GIF capture |
| [matplotlib](https://matplotlib.org/) | Learning curves and driver comparison plots |
| [pandas](https://pandas.pydata.org/) | Results and telemetry |

---

## 💡 Inspiration

The reward shaping approach — using different penalty and bonus structures to produce distinct driving personalities — is conceptually similar to how real F1 teams use simulation to explore different setup philosophies. Except their agents don't drive into walls quite as often.

---

*"We are checking the data. The agent is, uh... we are checking."*
