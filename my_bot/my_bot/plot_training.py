#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    log_file = "/home/ahlem/mobile-3d-lidar-sim-main/my_bot/tmp/dqn/training_log.csv"
    save_dir = "/home/ahlem/mobile-3d-lidar-sim-main/my_bot/tmp/dqn"

    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    data = pd.read_csv(log_file)
    episodes = data["episode"]
    reward = data["reward"]
    distance = data["final_distance"]
    steps = data["steps"]
    reached_goal = data["reached_goal"]
    best_reward = data["best_reward"]
    epsilon = data["epsilon"]

    plt.figure(figsize=(14,10))

    plt.subplot(3,2,1)
    plt.plot(episodes, reward, 'b-')
    plt.scatter(episodes[reached_goal==1], reward[reached_goal==1], color='green', s=50)
    plt.title("Reward")
    plt.grid(True)

    plt.subplot(3,2,2)
    plt.plot(episodes, distance, 'r-')
    plt.scatter(episodes[reached_goal==1], distance[reached_goal==1], color='green', s=50)
    plt.title("Distance")
    plt.grid(True)

    plt.subplot(3,2,3)
    plt.plot(episodes, steps, 'g-')
    plt.scatter(episodes[reached_goal==1], steps[reached_goal==1], color='blue', s=50)
    plt.title("Steps")
    plt.grid(True)

    plt.subplot(3,2,4)
    plt.plot(episodes, best_reward, 'purple')
    plt.title("Best Reward")
    plt.grid(True)

    plt.subplot(3,2,5)
    plt.plot(episodes, distance, 'orange')
    plt.title("Distance Progress")
    plt.grid(True)

    plt.subplot(3,2,6)
    plt.plot(episodes, epsilon, 'brown')
    plt.title("Epsilon-Greedy")
    plt.grid(True)
    
    loss = data["loss"]

    plt.subplot(4,2,7)
    plt.plot(episodes, loss, 'black')
    plt.title("Loss")
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    print(f"✅ Saved curves to {os.path.join(save_dir, 'training_curves.png')}")
    plt.show()

if __name__ == "__main__":
    main()
