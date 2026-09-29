#!/usr/bin/env python3
"""
test.py — Test du modèle DQN entraîné

Usage:
    ros2 run my_bot test
    ou directement: python3 test.py
"""

import rclpy
import torch
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_bot.robot_env import RobotEnv
from my_bot.dqn_agent import DQNAgent

# =========================================================
# CHEMINS
# =========================================================
BASE_DIR   = "/home/ahlem/mobile-3d-lidar-sim-main"
MODEL_PATH = os.path.join(BASE_DIR, "dqn_model.pth")
BEST_PATH  = os.path.join(BASE_DIR, "dqn_model_best.pth")
CSV_PATH   = os.path.join(BASE_DIR, "test_log.csv")
PLOT_PATH  = os.path.join(BASE_DIR, "test_plot.png")

# =========================================================
# GOAL — DOIT ÊTRE IDENTIQUE À train.py
# =========================================================
GOAL_X = 8.0    # ← même valeur que dans train.py
GOAL_Y = 8.0    # ← même valeur que dans train.py

# Seuil succès cohérent avec reward_goal = +10.0
SUCCESS_THRESHOLD = 9.5

# =========================================================
# NOMS DES ACTIONS (pour affichage)
# =========================================================
ACTION_NAMES = {
    0: "Avance vite  ",
    1: "Avance lent  ",
    2: "Tourne gauche",
    3: "Tourne droite",
    4: "Pivot gauche ",
    5: "Pivot droite ",
}


# =========================================================
# PLOT
# =========================================================
def plot_test(rewards, distances, actions_all, success_flags, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Test DQN — Navigation autonome", fontsize=14)

    # Reward par épisode
    colors = ["green" if s else "tomato" for s in success_flags]
    axes[0, 0].bar(range(len(rewards)), rewards, color=colors, edgecolor="white")
    axes[0, 0].set_title("Reward total par épisode (vert=succès, rouge=échec)")
    axes[0, 0].grid(True, alpha=0.3)

    # Distance finale
    axes[0, 1].plot(distances, marker="o", color="green")
    axes[0, 1].axhline(
        RobotEnv.GOAL_DIST, color="red",
        linestyle="--", label=f"seuil goal ({RobotEnv.GOAL_DIST}m)"
    )
    axes[0, 1].set_title("Distance finale au goal")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution des actions
    unique, counts = np.unique(actions_all, return_counts=True)
    labels = [ACTION_NAMES.get(int(u), str(u)) for u in unique]
    axes[1, 0].bar(labels, counts, color="steelblue", edgecolor="white")
    axes[1, 0].set_title("Distribution des actions")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # Succès / Échec
    axes[1, 1].bar(
        range(len(success_flags)),
        success_flags,
        color=["green" if s else "tomato" for s in success_flags],
        edgecolor="white",
    )
    axes[1, 1].set_title("Succès (vert) / Échec (rouge) par épisode")
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(["Échec", "Succès"])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[PLOT] Sauvegardé → {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    rclpy.init()

    env   = RobotEnv(goal_x=GOAL_X, goal_y=GOAL_Y)
    agent = DQNAgent(
        state_size  = RobotEnv.STATE_SIZE,
        action_size = RobotEnv.ACTION_SIZE,
    )

    # ── Charger le meilleur modèle disponible ────────────
    load_path = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
    if not os.path.exists(load_path):
        print(f"[TEST] ❌ Aucun modèle trouvé dans {BASE_DIR}")
        rclpy.shutdown()
        return

    agent.load(load_path)
    agent.q_net.eval()
    agent.epsilon = 0.0    # greedy pur — aucune exploration
    print(f"[TEST] Modèle chargé : {load_path}")
    print(f"[TEST] ε = 0.0 (greedy pur)")
    print(f"[TEST] Goal : ({GOAL_X}, {GOAL_Y})")

    # =========================================================
    # CONFIG TEST
    # =========================================================
    N_EPISODES = 20

    test_log      = []
    rewards       = []
    distances     = []
    actions_all   = []
    success_flags = []

    print(f"\n[TEST] {N_EPISODES} épisodes × {RobotEnv.MAX_STEPS} steps max\n")

    # ── torch.no_grad() global : pas de gradients en test ──
    with torch.no_grad():

        for ep in range(N_EPISODES):
            state        = env.reset()
            total_reward = 0.0
            success      = False
            ep_actions   = []

            print(f"\n{'='*55}")
            print(f"  ÉPISODE TEST {ep:2d}/{N_EPISODES}")
            print(f"{'='*55}")

            # ── Boucle steps ──────────────────────────────
            # NOTE : timeout géré par RobotEnv.MAX_STEPS (done=True)
            for step in range(RobotEnv.MAX_STEPS + 10):

                # Action greedy — argmax Q-valeurs
                state_t  = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_net(state_t)
                action   = int(torch.argmax(q_values).item())
                q_vals   = q_values.squeeze().cpu().numpy()

                next_state, reward, done = env.step(action, episode=ep)

                dist  = env.get_distance()
                angle = env.get_angle_to_goal()

                actions_all.append(action)
                ep_actions.append(action)
                total_reward += reward

                # Affichage détaillé step par step
                print(
                    f"  step {step:3d} | {ACTION_NAMES[action]} | "
                    f"R={reward:+6.3f} | D={dist:.3f}m | "
                    f"Q=[{', '.join(f'{v:+.2f}' for v in q_vals)}]"
                )

                test_log.append([ep, step, action, reward, dist, angle, int(done)])
                state = next_state

                if done:
                    # CORRIGÉ : seuil cohérent avec reward goal = +10.0
                    success = reward >= SUCCESS_THRESHOLD
                    status  = "🎯 GOAL !" if success else "💥 Collision / Timeout"
                    print(f"\n  → {status} après {step + 1} steps")
                    break

            # ── Fin épisode ───────────────────────────────
            env.stop()

            last_dist = env.get_distance()
            rewards.append(total_reward)
            distances.append(last_dist)
            success_flags.append(int(success))

            # Distribution actions de cet épisode
            unique_ep, counts_ep = np.unique(ep_actions, return_counts=True)
            action_summary = " | ".join(
                f"{ACTION_NAMES[int(u)].strip()}×{c}"
                for u, c in zip(unique_ep, counts_ep)
            )

            print(
                f"\n[EP {ep:2d}] {'✅ SUCCÈS' if success else '❌ ÉCHEC'}  "
                f"R={total_reward:+7.2f}  "
                f"D_finale={last_dist:.3f}m\n"
                f"         Actions : {action_summary}"
            )

    # =========================================================
    # RÉSUMÉ FINAL
    # =========================================================
    n_success = sum(success_flags)
    print(f"\n{'='*55}")
    print(f"  RÉSULTATS TEST ({N_EPISODES} épisodes)")
    print(f"  Succès     : {n_success}/{N_EPISODES} ({100 * n_success / N_EPISODES:.1f}%)")
    print(f"  Reward moy : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Dist moy   : {np.mean(distances):.3f}m")
    print(f"  Dist min   : {np.min(distances):.3f}m")
    print(f"{'='*55}\n")

    # =========================================================
    # SAUVEGARDE CSV
    # =========================================================
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "action", "reward",
                         "distance", "angle", "done"])
        writer.writerows(test_log)
    print(f"[TEST] CSV sauvegardé → {CSV_PATH}")

    # =========================================================
    # PLOTS
    # =========================================================
    plot_test(rewards, distances, actions_all, success_flags, PLOT_PATH)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
