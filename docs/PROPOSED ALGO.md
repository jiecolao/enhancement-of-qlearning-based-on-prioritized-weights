PROPOSED ALGO

ENHANCEMENTS:
1. Deep Q-learning
2. Decaying Epsilon
3. Double Deep Q-learning

4. EQLBPW collision/goal/distance weighting
5. Prioritized Experience Replay
6. Improved Importance Sampling
7. Changed reward shaping
8. Manhattan Distance

DDQN+Prioritized Replay+Path-Aware Priority+Importance Sampling

                Enhanced Q-Learning
                       │
          ┌────────────┴────────────┐
          │                         │
      Double DQN              Prioritized Replay
          │                         │
          │                    ┌────┴─────┐
          │                    │          │
          │                 TD Error   EQLBPW
          │                              │
          │                    ┌─────────┼─────────┐
          │                    │         │         │
          │                Collision   Goal   Distance
          │
          └──────────────┬─────────────────────────
                         │
                  ε-Greedy Exploration
                         │
                         ▼
                  Learned Policy
                         │
                         ▼
             Path/Success/Optimality
                    Evaluation


                 ENVIRONMENT
                     │
                     ▼
              State Representation
                     │
                     ▼
              EQLBPW DQN Agent
                     │
              ┌──────┴──────┐
              │             │
          Main DQN       Target DQN
              │             │
              └──────┬──────┘
                     │
                Double DQN
                     │
                     ▼
                TD Error δ
                     │
                     ▼
          ┌─────────────────────┐
          │ EQLBPW Priority     │
          │                     │
          │ TD Error            │
          │ Collision           │
          │ Goal                │
          │ Distance Progress   │
          └─────────┬───────────┘
                    │
                    ▼
          Prioritized Replay
                    │
                    ▼
       Importance-Sampling Weights
                    │
                    ▼
             Weighted Huber
                 Loss
                    │
                    ▼
             Network Update
                    │
                    ▼
          Periodic Target Sync
                    │
                    ▼
          Decaying ε-Greedy
                    │
                    ▼
            Next Environment
                Interaction

IMPROVED ALGORITHM
I. Initialization
1. Initialize Replay Memory D to capacity N.

2. Initialize Main Action-Value Neural Network Q with random weights θ.

3. Initialize Target Action-Value Neural Network Q' with weights θ' = θ.

4. Initialize Prioritized Experience Replay parameters:
    - B = batch size
    - α = priority exponent
    - βstart = initial importance-sampling exponent
    - βend= final importance-sampling exponent

5. Initialize EQLBPW priority-weight parameters:
    - wc= collision weight
    - wg= goal weight
    - wd = distance-progress weight

6. Initialize learning parameters:
    - η = learning rate
    - γ = discount factor

7. Initialize Exploration rate ε = 1.0, minimum εmin, and decay rate εdecay.

8. Initialize Step Count C = 0.

II. For Each Episode
1. Initialize starting state s.

2. Repeat until s is terminated or the maximum number of steps is reached.

    a. Generate a random number p ∈ [0, 1].
        if p < ε,
            a = random action.
        else,
            a = argmaxa' Q(s, a'; θ).

    b. Execute action a and observe:
        - reward r
        - next state s'
        - terminal status
        - collision indicator Ci
        - goal indicator Gi
        - distance progress Di

    c. Store transition in Replay Memory D with maximum priority.
        Transition =
        (s, a, r, s', done, collision, goal, distance_progress)
        Initial priority:
        pt = maxi < t pi
        If no previous priority exists:
        pt = 1.0

    d. If size of D >= B:

        d.1 Calculate prioritized sampling probability for each
            transition i.
            P(i) = piα / Σk pkα 

        Select a mini-batch of B transitions from D according
        to their prioritized sampling probabilities.

        d.2 Compute the importance-sampling weight for each
            sampled transition.
            wi = (N × P(i))(-β)

        Normalize the importance-sampling weights by dividing
        each weight by the maximum sampled weight.

        d.3 For each transition i in the mini-batch:

            d.3.1 If s'i is a terminal state:
                yi = ri
            else:
                Select the optimal next action using the Main Network:
                a'i = argmaxa' Q(s'i, a'; θ)
                Evaluate the selected action using the Target Network:
                Q_next = Q'(s'i, a'i; θ')
                yi = ri + γ × Q'(s'i , a'_i; θ')

            d.3.2 Compute the current Q-value:
                Qi = Q(si, ai; θ)

            d.3.3 Compute the TD error:
                δi = yi - Qi

            d.3.4 Calculate the EQLBPW priority weight.
                Wi =
                1
                + wc × Ci
                + wg × Gi
                + wd × max(Di, 0)

            where:
                Ci = collision indicator
                Gi = goal indicator
                Di = distance progress

            d.3.5 Calculate the updated EQLBPW priority:
                pi = (|δi| + εp) × Wi

            where εp is a small positive value used to prevent
            the priority from becoming zero.

            d.3.6 Update transition priority pi in Replay Memory D.

        d.4 Calculate the Huber loss for each sampled transition.
            Li = Huber(Qi, yi)

        Apply the importance-sampling weight:
            L = (1 / B) × Σi [wi× Li]

        Perform a gradient descent step on the Main Network Q
        using the weighted loss:
            θ ← θ - η × ∇_θ L

        Apply gradient clipping to limit excessively large
        gradients.

    e. Set s ← s'.

    f. Decay the exploration rate:
        ε ← max(εmin, ε × εdecay)

    g. Update the importance-sampling exponent according to
       training progress:
        β = βstart + progress × (βend - βstart)
        where:
            0 <= progress <= 1

    h. Every C steps, synchronize the Target Network:
        θ' ← θ

