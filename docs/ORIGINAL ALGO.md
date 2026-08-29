ORIGINAL ALGO

PROBLEMS:
1. State-space explosion
2. Local optima
3. Overestimation bias

ORIGINAL ALGORITHM:
I. Initialization:
1. Initialize Q-table, maximum samples (b), episodes (e), and reward (r).
2. Set super parameter β ∈ [0, 1].
3. Set learning rate α ∈ [0, 1].
4. Set discount factor γ [0, 1].
5. Set exploration rate ε ∈ [0, 1].

II. For each episode
1. Initialize status S.
2. Repeat until s is terminated.
	a. Select action a from state s using an ε-greedy policy:
		1 - ε,     a = arg⁡max⁡aQ(st,a). 
		ε,          a = random action.
	b. Execute action a then observe reward r and next state s′.
	c. Calculate sampling probability:
		Pj ∝ 1 / rank( j )
	d. Adjust learning rate:
		aj = a / (b * pj)^β
	e. Update the Q-value:
		qj = Qnow(sj , aj)
		qj+1 = maxaQnow(sj+1, a)
		yj = rj + γqj+1
		δj = qj - yj
		Qnew (sj , aj) ← (1 − α)Qnow(sj , aj) + αδj
	f. Set s ← s′
