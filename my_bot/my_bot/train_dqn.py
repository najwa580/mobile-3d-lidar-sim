#!/usr/bin/env python3
"""
train.py — Entraînement DQN pour navigation autonome ROS2

Usage:
    ros2 run my_bot train
    ou directement:
    python3 train.py
"""

import rclpy
import torch
import numpy as np
import random
import os
import json
import time
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from my_bot.robot_env import RobotEnv
from my_bot.dqn_agent import DQNAgent


# =========================================================
# CHEMINS
# =========================================================

BASE_DIR = "/home/ahlem/mobile-3d-lidar-sim-main"

MODEL_PATH = os.path.join(BASE_DIR, "dqn_model.pth")
LOG_PATH = os.path.join(BASE_DIR, "train_log.json")
PLOT_PATH = os.path.join(BASE_DIR, "train_plot.png")


# =========================================================
# REPRODUCTIBILITÉ
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# =========================================================
# CONFIGURATION ENTRAÎNEMENT
# =========================================================

EPISODES = 200
SAVE_EVERY = 10
PLOT_EVERY = 20
WARMUP_STEPS = 256

RESET_EPSILON = False

SUCCESS_THRESHOLD = 9.5


# =========================================================
# GOAL
# =========================================================

GOAL_X = -6.0
GOAL_Y = 6.0


# =========================================================
# CALCUL MOYENNE MOBILE
# =========================================================

def moving_average(values, window=10):

    values = np.array(values, dtype=np.float32)

    if len(values) == 0:
        return np.array([])

    result = []

    for i in range(len(values)):

        start = max(0, i - window + 1)

        result.append(
            np.mean(values[start:i + 1])
        )

    return np.array(result)


# =========================================================
# PLOT
# =========================================================

def plot_training(
    rewards,
    losses,
    distances,
    success_flags,
    collision_flags,
    steps_list,
    path
):

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    fig.suptitle(
        "Entraînement DQN — Navigation autonome",
        fontsize=15
    )


    # =====================================================
    # 1. REWARD MOYEN
    # =====================================================

    reward_ma = moving_average(rewards, window=10)

    axes[0, 0].plot(
        reward_ma,
        lw=2
    )

    axes[0, 0].set_title(
        "Reward moyen (fenêtre 10 épisodes)"
    )

    axes[0, 0].set_xlabel("Épisode")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 0].grid(True, alpha=0.3)


    # =====================================================
    # 2. LOSS
    # =====================================================

    non_zero_losses = [
        loss for loss in losses
        if loss > 0
    ]

    if non_zero_losses:

        axes[0, 1].plot(
            losses,
            alpha=0.8
        )

        axes[0, 1].set_yscale("log")

    axes[0, 1].set_title(
        "Loss moyenne par épisode"
    )

    axes[0, 1].set_xlabel("Épisode")
    axes[0, 1].set_ylabel("Loss")

    axes[0, 1].grid(True, alpha=0.3)


    # =====================================================
    # 3. DISTANCE FINALE
    # =====================================================

    distance_ma = moving_average(
        distances,
        window=10
    )

    axes[0, 2].plot(
        distance_ma,
        lw=2
    )

    axes[0, 2].axhline(
        RobotEnv.GOAL_DIST,
        linestyle="--",
        label=f"Seuil goal ({RobotEnv.GOAL_DIST} m)"
    )

    axes[0, 2].set_title(
        "Distance finale moyenne au goal"
    )

    axes[0, 2].set_xlabel("Épisode")
    axes[0, 2].set_ylabel("Distance (m)")

    axes[0, 2].legend()

    axes[0, 2].grid(True, alpha=0.3)


    # =====================================================
    # 4. TAUX DE SUCCÈS
    # =====================================================

    success_rate = moving_average(
        success_flags,
        window=20
    )

    axes[1, 0].plot(
        success_rate * 100,
        lw=2
    )

    axes[1, 0].set_title(
        "Taux de succès (fenêtre 20 épisodes)"
    )

    axes[1, 0].set_xlabel("Épisode")
    axes[1, 0].set_ylabel("Succès (%)")

    axes[1, 0].set_ylim(0, 105)

    axes[1, 0].grid(True, alpha=0.3)


    # =====================================================
    # 5. TAUX DE COLLISION
    # =====================================================

    collision_rate = moving_average(
        collision_flags,
        window=20
    )

    axes[1, 1].plot(
        collision_rate * 100,
        lw=2
    )

    axes[1, 1].set_title(
        "Taux de collision (fenêtre 20 épisodes)"
    )

    axes[1, 1].set_xlabel("Épisode")
    axes[1, 1].set_ylabel("Collisions (%)")

    axes[1, 1].set_ylim(0, 105)

    axes[1, 1].grid(True, alpha=0.3)


    # =====================================================
    # 6. NOMBRE DE PAS
    # =====================================================

    steps_ma = moving_average(
        steps_list,
        window=10
    )

    axes[1, 2].plot(
        steps_ma,
        lw=2
    )

    axes[1, 2].set_title(
        "Nombre moyen de pas par épisode"
    )

    axes[1, 2].set_xlabel("Épisode")
    axes[1, 2].set_ylabel("Nombre de pas")

    axes[1, 2].grid(True, alpha=0.3)


    # =====================================================
    # SAUVEGARDE
    # =====================================================

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    print(
        f"[PLOT] Sauvegardé → {path}"
    )


# =========================================================
# WARM-UP
# =========================================================

def warmup(env, agent, n_steps):

    print(
        f"[WARMUP] Remplissage buffer "
        f"({n_steps} steps aléatoires)…"
    )

    state = env.reset()

    filled = 0

    while filled < n_steps:

        action = random.randint(
            0,
            RobotEnv.ACTION_SIZE - 1
        )

        next_state, reward, done = env.step(
            action,
            episode=-1
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            float(done)
        )

        state = next_state

        filled += 1

        if done:
            state = env.reset()

    print(
        f"[WARMUP] Buffer rempli avec "
        f"{filled} transitions ✓"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    rclpy.init()


    # =====================================================
    # ENVIRONNEMENT ET AGENT
    # =====================================================

    env = RobotEnv(
        goal_x=GOAL_X,
        goal_y=GOAL_Y
    )

    agent = DQNAgent(
        state_size=RobotEnv.STATE_SIZE,
        action_size=RobotEnv.ACTION_SIZE
    )


    print(
        f"[TRAIN] State size  : "
        f"{RobotEnv.STATE_SIZE}"
    )

    print(
        f"[TRAIN] Action size : "
        f"{RobotEnv.ACTION_SIZE}"
    )

    print(
        f"[TRAIN] Goal        : "
        f"({GOAL_X}, {GOAL_Y})"
    )

    print(
        f"[TRAIN] Episodes    : "
        f"{EPISODES}"
    )


    # =====================================================
    # CHARGEMENT MODELE
    # =====================================================

    if os.path.exists(MODEL_PATH):

        agent.load(MODEL_PATH)

        print(
            f"[TRAIN] Modèle chargé depuis "
            f"{MODEL_PATH}"
        )

        if RESET_EPSILON:

            agent.epsilon = 1.0

            print(
                "[TRAIN] RESET_EPSILON=True "
                "→ epsilon forcé à 1.0"
            )

    else:

        print(
            "[TRAIN] Pas de modèle existant "
            "— démarrage de zéro"
        )


    # =====================================================
    # WARM-UP
    # =====================================================

    if not os.path.exists(MODEL_PATH):

        warmup(
            env,
            agent,
            WARMUP_STEPS
        )


    # =====================================================
    # LOGS
    # =====================================================

    rewards = []

    losses = []

    distances = []

    success_flags = []

    collision_flags = []

    steps_list = []

    navigation_times = []

    path_lengths = []


    best_reward = float("-inf")


    # =====================================================
    # BOUCLE PRINCIPALE
    # =====================================================

    for ep in range(EPISODES):


        # -----------------------------------------------
        # Début épisode
        # -----------------------------------------------

        t_start = time.time()

        state = env.reset()


        total_reward = 0.0

        total_loss = 0.0

        loss_count = 0


        success = False

        collision = False

        episode_steps = 0


        print(
            f"\n{'=' * 55}"
        )

        print(
            f"EPISODE {ep:3d}/{EPISODES}"
        )

        print(
            f"{'=' * 55}"
        )


        # =================================================
        # BOUCLE STEPS
        # =================================================

        for step in range(
            RobotEnv.MAX_STEPS + 10
        ):

            action = agent.act(state)


            next_state, reward, done = env.step(
                action,
                episode=ep
            )


            agent.remember(
                state,
                action,
                reward,
                next_state,
                float(done)
            )


            loss = agent.replay()


            if loss > 0:

                total_loss += loss

                loss_count += 1


            state = next_state


            total_reward += reward


            episode_steps += 1


            # ---------------------------------------------
            # FIN EPISODE
            # ---------------------------------------------

            if done:

                # Succès
                success = (
                    reward >= SUCCESS_THRESHOLD
                )


                # Collision
                #
                # RobotEnv doit définir env.collision
                #
                collision = bool(
                    getattr(
                        env,
                        "collision",
                        False
                    )
                )


                break


        # =================================================
        # FIN EPISODE
        # =================================================

        agent.decay_epsilon()

        env.stop()


        # Distance finale
        last_dist = env.get_distance()


        # Loss moyenne
        avg_loss = (
            total_loss / loss_count
            if loss_count > 0
            else 0.0
        )


        # Temps navigation
        ep_time = (
            time.time() - t_start
        )


        # =================================================
        # LONGUEUR TRAJECTOIRE
        # =================================================
        #
        # RobotEnv doit fournir env.path
        # sous forme [(x1,y1), (x2,y2), ...]
        #
        # =================================================

        path_length = 0.0


        if (
            hasattr(env, "path")
            and len(env.path) >= 2
        ):

            for i in range(
                1,
                len(env.path)
            ):

                x1, y1 = env.path[i - 1]

                x2, y2 = env.path[i]


                path_length += np.hypot(
                    x2 - x1,
                    y2 - y1
                )


        # =================================================
        # STOCKAGE METRIQUES
        # =================================================

        rewards.append(
            float(total_reward)
        )

        losses.append(
            float(avg_loss)
        )

        distances.append(
            float(last_dist)
        )

        success_flags.append(
            int(success)
        )

        collision_flags.append(
            int(collision)
        )

        steps_list.append(
            int(episode_steps)
        )

        navigation_times.append(
            float(ep_time)
        )

        path_lengths.append(
            float(path_length)
        )


        # =================================================
        # AFFICHAGE
        # =================================================

        if success:

            status = "🎯 SUCCÈS"

        elif collision:

            status = "💥 COLLISION"

        else:

            status = "⏱ TIMEOUT"


        print(
            f"\n[EP {ep:3d}] {status}"
        )

        print(
            f"Reward       = {total_reward:+.2f}"
        )

        print(
            f"Distance     = {last_dist:.2f} m"
        )

        print(
            f"Steps        = {episode_steps}"
        )

        print(
            f"Temps        = {ep_time:.2f} s"
        )

        print(
            f"Trajectoire  = {path_length:.2f} m"
        )

        print(
            f"Loss         = {avg_loss:.5f}"
        )


        # =================================================
        # SAUVEGARDE PERIODIQUE
        # =================================================

        if (
            (ep + 1) % SAVE_EVERY == 0
        ):

            agent.save(
                MODEL_PATH
            )


            log = {

                "seed": SEED,

                "rewards":
                    rewards,

                "losses":
                    losses,

                "distances":
                    distances,

                "success_flags":
                    success_flags,

                "collision_flags":
                    collision_flags,

                "steps":
                    steps_list,

                "navigation_times":
                    navigation_times,

                "path_lengths":
                    path_lengths,

                "episodes_done":
                    ep + 1
            }


            with open(
                LOG_PATH,
                "w"
            ) as f:

                json.dump(
                    log,
                    f,
                    indent=4
                )


            print(
                f"[TRAIN] Log sauvegardé → "
                f"{LOG_PATH}"
            )


        # =================================================
        # MEILLEUR MODELE
        # =================================================

        if (
            total_reward > best_reward
        ):

            best_reward = total_reward


            best_path = MODEL_PATH.replace(
                ".pth",
                "_best.pth"
            )


            agent.save(
                best_path
            )


            print(
                f"[TRAIN] 🏆 Nouveau meilleur reward : "
                f"{best_reward:.2f}"
            )


        # =================================================
        # PLOT PERIODIQUE
        # =================================================

        if (
            (ep + 1) % PLOT_EVERY == 0
        ):

            plot_training(

                rewards,
                losses,
                distances,

                success_flags,
                collision_flags,

                steps_list,

                PLOT_PATH
            )


    # =====================================================
    # FIN ENTRAINEMENT
    # =====================================================

    env.stop()


    env.save_logs(
        os.path.join(
            BASE_DIR,
            "path.csv"
        )
    )


    agent.save(
        MODEL_PATH
    )


    plot_training(

        rewards,
        losses,
        distances,

        success_flags,
        collision_flags,

        steps_list,

        PLOT_PATH
    )


    # =====================================================
    # STATISTIQUES FINALES
    # =====================================================

    n_success = sum(
        success_flags
    )

    n_collision = sum(
        collision_flags
    )


    print(
        f"\n{'=' * 55}"
    )

    print(
        "ENTRAÎNEMENT TERMINÉ"
    )

    print(
        f"Seed : {SEED}"
    )

    print(
        f"Succès : "
        f"{n_success}/{EPISODES} "
        f"({100 * n_success / EPISODES:.1f}%)"
    )

    print(
        f"Collisions : "
        f"{n_collision}/{EPISODES} "
        f"({100 * n_collision / EPISODES:.1f}%)"
    )

    print(
        f"Reward moyen : "
        f"{np.mean(rewards):.2f}"
    )

    print(
        f"Pas moyens : "
        f"{np.mean(steps_list):.1f}"
    )

    print(
        f"Temps moyen : "
        f"{np.mean(navigation_times):.2f} s"
    )

    print(
        f"Longueur moyenne : "
        f"{np.mean(path_lengths):.2f} m"
    )

    print(
        f"Distance finale moyenne : "
        f"{np.mean(distances):.2f} m"
    )

    print(
        f"Meilleur reward : "
        f"{best_reward:.2f}"
    )

    print(
        f"{'=' * 55}"
    )


    rclpy.shutdown()


if __name__ == "__main__":

    main()
