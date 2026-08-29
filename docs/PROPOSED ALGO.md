PROPOSED ALGO

ENHANCEMENTS:
1. Deep Q-learning
2. Decaying Epsilon
3. Double Deep Q-learning

IMPROVED ALGORITHM
I. Initialization
	1. Initialize Replay Memory D to capacity N.
	2. Initialize Main Action-Value Neural Network Q with random weights θ.
	3. Initialize Target Action-Value Neural Network Q' with weights θ ' = θ.
	4. Initialize Prioritized Weight parameters: 
		b (batch size), β (super parameter/beta), α (learning rate), e (episodes), and rewards (r). 
	5. Initialize Discount factor γ.
	6. Initialize Exploration rate ε = 1.0, minimum εmin and decay rate εdecay
    7. Initialize Step Count C = 0.

II. For each episode
	1. Initialize starting status s.
	2. Repeat until s is terminated.
		a. Generate a random number p ∈ [0, 1]
			if p < e,		a = random action.
			else,			a = argmaxa'Q(s, a'; θ).
		b. Execute action a then observe reward r and next status s′.
		c. Store transition (s, a, r, s') in Replay Memory D with maximal priority.
			pt = maxi<t pi
		d. If size of D > b
			d.1 Randomly sample a mini-batch of b transitions (sj, aj, rj, s'j) from D based on 			prioritized probability Pj.
				Pj ∝ 1 / rank( j )
			d.2 Compute importance-sampling weight w for each transition.
				wj = (1 / N * P( j ))^β
			d.3 For each transition j in mini-batch
				d.3.1 If s'j is a terminal state:
						Target value yj = rj
					else:
						Select optimal next action using Main Net:
							a'j = argmaxa'Q(s'j, a'; θ)
						Evaluate Q-value using Target Net:  			
							yj = rj + γQ'(s'j, a'j; θ')
				d.3.2. Compute TD error
						δj = yj - Q(sj, aj; θ)
					Update transition priority pj = |δj| in memory D
			d.4 Perform a gradient descent step on Main Network Q using weights wj and 
			loss (yj - Q(sj, aj; θ))2
				α = α / (b * pj)^β
		e. Set s ← s′.
		f. Decay exploration rate: ε ← max(εmin, ε * εdecay)
		g. Every C steps, synchronize Target Network: θ' ← θ

