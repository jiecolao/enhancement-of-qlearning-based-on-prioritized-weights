# Further Enhancement of Improved Q-learning Algorithm Applied to Dynamic Obstacle Avoidance and Path Planning

This study further enhances the study of Chunlei Wang, Xiao Yang, and He Li named _Improved Q-Learning Applied to Dynamic Obstacle Avoidance and Path Planning_


## Statement of the Problem

1. State-space explosion
```
"As the scale of the problem increases, the Q-table in the Q-learning algorithm will also expand, increasing the algorithm’s complexity."
```
2. Local optima entrapment
```
"The Q-learning algorithm may produce a locally optimal solution rather than a globally optimal one, resulting in the agent not obtaining a higher reward."
```
3. Overestimation Bias
```
"Q-learning has the problem of overestimation, which makes it impossible to choose the optimal action."
```

## Objectives of the Study
1. Deep Q-Learning
2. Advanced exploration strategy 
3. Double DQL

Wang, C., et al. (2022). Improved Q-Learning Applied to Dynamic Obstacle Avoidance and Path Planning. https://ieeexplore.ieee.org/abstract/document/9870811

## HOW TO RUN:
- cd src
- streamlit run app.py
- python -m EQLBPW.simulator
- python -m QLBPW.simulator

The Streamlit dashboard uses the sidebar to switch between EQLBPW and QLBPW.
Install the dependencies from `requirements.txt`, including Streamlit and PyTorch
for the EQLBPW section.