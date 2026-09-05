import random

class BanditArm:
    """Represents a single arm of a multi-armed bandit, providing rewards."""
    def __init__(self, true_mean_reward):
        self.true_mean_reward = true_mean_reward

    def pull(self):
        # Simulate a binary reward (1 for success, 0 for failure)
        # The true_mean_reward acts as the probability of success.
        return 1 if random.random() < self.true_mean_reward else 0

class EpsilonGreedyAgent:
    """
    An agent that learns to choose the best bandit arm using an epsilon-greedy strategy.
    This demonstrates trial-and-error learning and self-improvement.
    """
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon  # Probability of exploration (trying a random arm)
        self.q_values = [0.0] * num_arms  # Estimated value (average reward) for each arm
        self.n_pulls = [0] * num_arms    # Number of times each arm has been pulled

    def choose_action(self):
        # This is where the agent decides whether to explore or exploit.
        # Exploration: try a random arm to gather new information.
        # Exploitation: choose the arm currently believed to be the best.
        if random.random() < self.epsilon:
            return random.randrange(self.num_arms)
        else:
            # If multiple arms have the same max Q-value, pick the first one.
            return self.q_values.index(max(self.q_values))

    def update_q_value(self, action, reward):
        # The agent learns from experience by updating its estimate for the chosen arm.
        # This is the 'self-improvement' step based on observed outcomes.
        self.n_pulls[action] += 1
        # Incremental update rule for the average reward (Q-value)
        self.q_values[action] += (reward - self.q_values[action]) / self.n_pulls[action]

def run_simulation(num_arms, true_means, num_steps, epsilon=0.1):
    """Simulates the agent interacting with the bandit environment."""
    arms = [BanditArm(mean) for mean in true_means]
    agent = EpsilonGreedyAgent(num_arms, epsilon)

    total_reward = 0

    print(f"Simulation started with {num_arms} arms. True reward probabilities: {true_means}")
    print(f"Epsilon (exploration rate): {epsilon}")
    print("-" * 40)

    for step in range(num_steps):
        action = agent.choose_action() # Agent decides which arm to pull (trial)
        reward = arms[action].pull()   # Agent interacts with the environment and gets a reward
        agent.update_q_value(action, reward) # Agent learns from the observed reward (error/success)

        total_reward += reward

        if (step + 1) % (num_steps // 10) == 0 or step == 0:
            print(f"Step {step+1}/{num_steps}: Chosen Arm {action}, Reward {reward}")
            print(f"  Current Q-values (estimates): {[f'{q:.2f}' for q in agent.q_values]}")
            print(f"  Total reward so far: {total_reward}")
            print("-" * 20)

    print("-" * 40)
    print("Simulation finished.")
    print(f"Final Q-values (estimates): {[f'{q:.2f}' for q in agent.q_values]}")
    print(f"True reward probabilities: {true_means}")
    print(f"Total reward collected: {total_reward}")
    print(f"Optimal total reward (if agent always chose best arm): {max(true_means) * num_steps}")
    print(f"The agent's performance improves by learning to pick arms with higher estimated Q-values over time.")

if __name__ == "__main__":
    # Define the bandit problem: 3 arms with different true reward probabilities
    true_reward_probabilities = [0.2, 0.8, 0.5] # Arm 0 (20%), Arm 1 (80%), Arm 2 (50%)
    num_bandits = len(true_reward_probabilities)
    simulation_steps = 1000 # Number of times the agent pulls an arm

    run_simulation(num_bandits, true_reward_probabilities, simulation_steps, epsilon=0.1)
