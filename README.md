# Comp-6600-Final-Project
Thomas Richards and Muhammad Taimoor Hassan COMP 6600/7700 Project, Fall 2025 Instructor: Prof. Sathyanarayanan Aakur

## Project Overview
This project implements a reinforcement-learning agent to play Rock–Paper–Scissors (RPS) using Q-Learning. 
We evaluate the agent against multiple fixed-strategy opponents and analyze its performance using 
quantitative metrics such as win rate, histograms, and boxplots.

## File Structure
- main.py — runs training, evaluation, and produces final plots
- plots/
    - winrate_histogram.png
    - winrate_boxplot.png
- results/
    - stats_results.txt
- README.md — project documentation

## Installation
Create a Python 3.10+ environment and install the required libraries:

pip install -r requirements.txt

Main libraries:
- numpy
- matplotlib
- random

## How to Run
Simply run:

python main.py

This will:
1. Train the Q-Learning agent
2. Evaluate it against all benchmark opponent strategies
3. Generate the histogram and boxplot inside the /plots folder

## Q-Learning Summary
The Q-Learning agent maintains a Q-table that maps (state, action) → expected reward.
On each round:
- It chooses an action using ε-greedy exploration
- It receives a reward (+1 win, 0 tie, -1 loss)
- It updates its Q-values using the Bellman equation

The agent learns long-term patterns in opponents’ repeated behaviors.

## Opponent Strategies
We evaluate the agent against:
- Random: picks R/P/S uniformly
- RRPPSS: deterministic repeating sequence
- Beat-Last: always plays the move that would beat the agent's previous move
- Copycat: copies the agent’s previous move

## Experimental Setup
- Episodes per match: 1,000
- Opponents tested: 4
- Trials per opponent: 30
- Metric: win rate (wins / total rounds)

Results are visualized using histograms and boxplots.

## Results
The Q-Learning agent consistently outperforms the baseline strategies.

### Win Rate Histogram
Shows distribution of performance per opponent.

### Win Rate Boxplot
Illustrates variability across trials. 
The RRPPSS baseline has the *narrowest* box, indicating very stable win rates.

## Conclusion
The Q-Learning agent successfully adapts to predictable and semi-predictable strategies. 
It struggles more against adaptive opponents like Beat-Last, but still performs competitively.

This demonstrates how reinforcement learning can succeed even in simple adversarial environments.
