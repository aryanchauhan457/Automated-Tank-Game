import pygame
import numpy as np
import math
import random
import os

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Automated Tank Game")

# Load assets
tank_img = pygame.image.load("tank.png")
bullet_img = pygame.image.load("bullet.png")
target_img = pygame.image.load("target.png")
pygame.display.set_icon(pygame.image.load("icon.png"))

# Q-Learning setup
angle_bins = 36
distance_bins = 10
actions = ["left", "right", "fire"]
q_table_path = "q_table.npy"

# Load or initialize Q-table
if os.path.exists(q_table_path):
    q_table = np.load(q_table_path)
    print("[INFO] Loaded existing Q-table.")
else:
    q_table = np.random.uniform(low=-1, high=1, size=(angle_bins, distance_bins, len(actions)))
    print("[INFO] Created new Q-table.")

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

# Game constants
tank_pos = pygame.Rect(370, 500, 60, 60)
angle = 0
bullet_status = "ready"
bullet_pos = pygame.Rect(390, 490, 20, 80)
b_x, b_y = 390, 490
x_inc, y_inc = 0, 0
b_angle = 0
bullet_speed = 5

def draw_tank(a):
    rotated = pygame.transform.rotate(tank_img, a)
    rect = rotated.get_rect(center=tank_pos.center)
    screen.blit(rotated, rect)
    return rect

def draw_bullet(a, pos):
    rotated = pygame.transform.rotate(bullet_img, a)
    rect = rotated.get_rect(center=pos.center)
    screen.blit(rotated, rect)

def draw_target(target_pos):
    screen.blit(target_img, target_pos)

def calculate_distance(pos1, pos2):
    return math.hypot(pos1.centerx - pos2.centerx, pos1.centery - pos2.centery)

def get_state(angle, distance):
    angle_bin = int(angle % 360 // 10)
    dist_bin = min(int(distance // 100), distance_bins - 1)
    return angle_bin, dist_bin

def select_action(state):
    if np.random.rand() < epsilon:
        return random.randint(0, len(actions) - 1)
    return np.argmax(q_table[state[0], state[1]])

def update_q_table(prev_state, action, reward, new_state):
    max_future = np.max(q_table[new_state[0], new_state[1]])
    current_q = q_table[prev_state[0], prev_state[1], action]
    q_table[prev_state[0], prev_state[1], action] = current_q + learning_rate * (reward + discount_factor * max_future - current_q)

# --- Main Loop ---
episodes = 1
for episode in range(episodes):
    angle = 0
    bullet_status = "ready"
    target_pos = pygame.Rect(random.randint(20, 730), random.randint(20, 300), 50, 50)
    done = False

    while not done:
        pygame.time.delay(10)
        screen.fill((40, 100, 40))
        draw_target(target_pos)

        distance = calculate_distance(tank_pos, target_pos)
        state = get_state(angle, distance)

        action = select_action(state)

        reward = -1  # Small penalty for each time step

        # --- Action Handling ---
        if actions[action] == "left":
            angle = (angle + 10) % 360

        elif actions[action] == "right":
            angle = (angle - 10) % 360

        elif actions[action] == "fire" and bullet_status == "ready":
            b_angle = angle
            bullet_status = "fire"
            b_x, b_y = 390, 490
            x_inc = -math.sin(math.radians(b_angle)) * bullet_speed
            y_inc = -math.cos(math.radians(b_angle)) * bullet_speed
            bullet_pos = pygame.Rect(b_x, b_y, 20, 80)

        # --- Bullet Movement ---
        if bullet_status == "fire":
            b_x += x_inc
            b_y += y_inc
            bullet_pos = pygame.Rect(b_x, b_y, 20, 80)
            draw_bullet(b_angle, bullet_pos)

            if b_x < 0 or b_x > 800 or b_y < 0 or b_y > 600:
                reward = -10
                bullet_status = "ready"

            if calculate_distance(bullet_pos, target_pos) < 50:
                reward = 100
                bullet_status = "ready"
                done = True

        # --- Q-Learning Update ---
        new_distance = calculate_distance(tank_pos, target_pos)
        new_state = get_state(angle, new_distance)
        update_q_table(state, action, reward, new_state)

        draw_tank(angle)
        pygame.display.update()

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# --- Save Q-table ---
np.save(q_table_path, q_table)
print(f"[INFO] Q-table saved to {q_table_path}")

pygame.quit()
