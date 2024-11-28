import os
import sys
import random

import numpy as np

import time
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
SAVE_DIR = "game_states"
import pickle


class SnakeGame:
    def __init__(self, length, is_grow, formation='空', seed=0, board_size=21, silent_mode=True, random_states=[]):
        self.board_size = board_size
        self.grid_size = self.board_size ** 2
        self.cell_size = 40
        self.width = self.height = self.board_size * self.cell_size

        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size
        self.display_height = self.height + 2 * self.border_size + 40
        self.is_grow = is_grow

        self.silent_mode = silent_mode
        if not silent_mode:
            pygame.init()
            pygame.display.set_caption("Snake Game")
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            self.font = pygame.font.Font(None, 36)

            # Load sound effects
            # pygame.mixer.init()
            # self.sound_eat = pygame.mixer.Sound("sound/eat.wav")
            # self.sound_game_over = pygame.mixer.Sound("sound/game_over.wav")
            # self.sound_victory = pygame.mixer.Sound("sound/victory.wav")
        else:
            self.screen = None
            self.font = None

        self.snake = None
        self.non_snake = None

        self.direction = None
        self.score = 0
        self.food = None
        self.seed_value = seed
        self.length = length
        self.random_states = random_states
        self.formation = formation

        random.seed(seed) # Set random seed.
        
        self.reset()

    def reset(self):
        length = self.length
        if self.formation == '隨':
            names = ['東', '南', '西', '北', '天', '地']
            formation = random.choice(names)
        elif self.formation == '終焉':  # in this formation, length must be a states list
            p = random.uniform(0, 1)
            if p < 0.2:  # 20% to play 6dau '隨' with length between 30 to 150
                length = random.randint(50, 150)
                names = ['東', '南', '西', '北', '天', '地']
                formation = random.choice(names)
            elif 0.2 <= p and p < 0.4:  # 20% to play len=3
                self.snake = self._get_init_snake(3)
                formation = self.formation
            else:  # 60% to load a state
                formation = '空'
        else:
            formation = self.formation
            
        if formation == '空':
            if length == 'random':
                length = random.randint(3, 300 + len(self.random_states))
                if length <= 300:
                    self.snake = self._get_init_snake(length)
                else:
                    load_state_name = random.choice(self.random_states)
                    self.load_state(load_state_name)
            elif isinstance(length, list):
                load_state_name = random.choice(self.length)
                self.load_state(load_state_name)
            else:
                self.snake = self._get_init_snake(length)
        if isinstance(length, int):
            if formation == '東':
                self.snake = self._get_init_snake_east(length)
            if formation == '南':
                self.snake = self._get_init_snake_south(length)
            if formation == '西':
                self.snake = self._get_init_snake_west(length)
            if formation == '北':
                self.snake = self._get_init_snake_north(length)
            if formation == '天':
                self.snake = self._get_init_snake_sky(length)
            if formation == '地':
                self.snake = self._get_init_snake_ground(length)
        
        # self.snake = [(self.board_size // 2 + i, self.board_size // 2) for i in range(1, -2, -1)] # Initialize the snake with three cells in (row, column) format.
        self.non_snake = set([(row, col) for row in range(self.board_size) for col in range(self.board_size) if (row, col) not in self.snake]) # Initialize the non-snake cells.
        self.direction = "WAITING"
        if self.snake[0][0] - self.snake[1][0] == 1:  # if head is lower than snake's first body
            self.direction = "DOWN"
        elif self.snake[0][0] - self.snake[1][0] == -1:  # if head is higher than snake's first body
            self.direction = "UP"
        elif self.snake[0][1] - self.snake[1][1] == 1:  # if head is righter than snake's first body
            self.direction = "RIGHT"
        elif self.snake[0][1] - self.snake[1][1] == -1:  # if head is lefter than snake's first body
            self.direction = "LEFT"
        self.food = self._generate_food()
        self.score = 0
        random.seed()

    def save_state(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        state = {
            "snake": self.snake,
            "direction": self.direction,
            "food": self.food,
            "score": self.score
        }
        with open(SAVE_DIR + '/len' + str(len(self.snake)) + '_state_' + time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())) + '.obj', 'wb') as file:
            pickle.dump(state, file)

    def load_state(self, file_name):
        with open(SAVE_DIR + "/" + file_name, 'rb') as file:
            state = pickle.load(file)
        self.snake = state['snake']
        self.non_snake = set([(row, col) for row in range(self.board_size) for col in range(self.board_size) if (row, col) not in self.snake])
        self.direction = state['direction']
        self.food = state['food']
        self.score = state['score']

    def step(self, action):
        self._update_direction(action) # Update direction based on action.

        # Move snake based on current action.
        row, col = self.snake[0]
        if self.direction == "UP":
            row -= 1
        elif self.direction == "DOWN":
            row += 1
        elif self.direction == "LEFT":
            col -= 1
        elif self.direction == "RIGHT":
            col += 1

        # Check if snake eats food.
        if (row, col) == self.food: # If snake eats food, it won't pop the last cell. The food grid will be taken by snake later, no need to update board vacancy matrix.
            food_obtained = True
            self.score += 10 # Add 10 points to the score when food is eaten.
            # if not self.silent_mode:
            #     self.sound_eat.play()
        else:
            food_obtained = False
            if self.is_grow:
                self.non_snake.add(self.snake.pop()) # Pop the last cell of the snake and add it to the non-snake set.
        if not self.is_grow:
            self.non_snake.add(self.snake.pop())
        # Check if snake collided with itself or the wall
        done = (
            (row, col) in self.snake
            or row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        )

        if not done:
            self.snake.insert(0, (row, col))
            self.non_snake.remove((row, col))

        # else: # If game is over and the game is not in silent mode, play game over sound effect.
            # if not self.silent_mode:
            #     if len(self.snake) < self.grid_size:
            #         self.sound_game_over.play()
            #     else:
            #         self.sound_victory.play()

        # Add new food after snake movement completes.
        if food_obtained:
            self.food = self._generate_food()

        info ={
            "snake_size": len(self.snake),
            "snake_head_pos": np.array(self.snake[0]),
            "prev_snake_head_pos": np.array(self.snake[1]),
            "food_pos": np.array(self.food),
            "food_obtained": food_obtained
        }

        return done, info

    # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    def _update_direction(self, action):
        if action == 0:
            if self.direction != "DOWN":
                self.direction = "UP"
        elif action == 1:
            if self.direction != "RIGHT":
                self.direction = "LEFT"
        elif action == 2:
            if self.direction != "LEFT":
                self.direction = "RIGHT"
        elif action == 3:
            if self.direction != "UP":
                self.direction = "DOWN"
        # Swich Case is supported in Python 3.10+

    def _generate_food(self):
        if len(self.non_snake) > 0:
            food = random.sample(self.non_snake, 1)[0]
        else: # If the snake occupies the entire board, no need to generate new food and just default to (0, 0).
            food = (0, 0)
        return food
    
    def _get_init_snake(self, length):
        if length <= self.board_size * self.board_size // 2:
            reversed_snake = [(0, 0)]
            sector = 1
            finish = False
            for i in range(1, self.board_size):
                for j in range(1, i + 1):
                    top_element = reversed_snake[len(reversed_snake) - 1]
                    reversed_snake.append((top_element[0] + sector, top_element[1]))
                    if len(reversed_snake) >= length:
                        finish = True
                        break
                if finish:
                    break

                for j in range(1, i + 1):
                    top_element = reversed_snake[len(reversed_snake) - 1]
                    reversed_snake.append((top_element[0], top_element[1] + sector))
                    if len(reversed_snake) >= length:
                        finish = True
                        break
                if finish:
                    break
                sector *= -1
            reversed_snake.reverse()
            result_snake = reversed_snake  # rename the reversed snake

            for i in range(len(result_snake)):
                result_snake[i] = (result_snake[i][0] + self.board_size // 2, result_snake[i][1] + self.board_size // 2)  # shift the snake to the middle
        else:
            result_snake = []
            finish = False
            for i in range(self.board_size):
                for j in range(self.board_size - 2):
                    if i % 2 == 0:
                        result_snake.append((i, self.board_size - 3 - j))
                    else:
                        result_snake.append((i, j))
                    if len(result_snake) >= length:
                        finish = True
                        break
                if finish:
                    break
        
        return result_snake
    
    def _get_init_snake_north(self, length):
        result_snake = []
        finish = False
        for i in range(self.board_size):
            for j in range(self.board_size - 2):
                if i % 2 == 0:
                    result_snake.append((i, self.board_size - 3 - j))
                else:
                    result_snake.append((i, j))
                if len(result_snake) >= length:
                    finish = True
                    break
            if finish:
                break
        
        return result_snake
    
    def _get_init_snake_east(self, length):
        result_snake = []
        finish = False
        for i in range(self.board_size):
            for j in range(self.board_size - 2):
                if i % 2 == 0:
                    result_snake.append((self.board_size - 3 - j, self.board_size - 1 - i))
                else:
                    result_snake.append((j, self.board_size - 1 - i))
                if len(result_snake) >= length:
                    finish = True
                    break
            if finish:
                break
        return result_snake
    
    def _get_init_snake_south(self, length):
        result_snake = []
        finish = False
        for i in range(self.board_size):
            for j in range(2, self.board_size):
                if i % 2 == 0:
                    result_snake.append((self.board_size - 1 - i, j))
                else:
                    result_snake.append((self.board_size - 1 - i, self.board_size + 1 - j))
                if len(result_snake) >= length:
                    finish = True
                    break
            if finish:
                break
        return result_snake
    
    def _get_init_snake_west(self, length):
        result_snake = []
        finish = False
        for i in range(self.board_size):
            for j in range(2, self.board_size):
                if i % 2 == 0:
                    result_snake.append((j, i))
                else:
                    result_snake.append((self.board_size + 1 - j, i))
                if len(result_snake) >= length:
                    finish = True
                    break
            if finish:
                break
        return result_snake
    
    def _get_init_snake_sky(self, length):
        reversed_snake = [(0, 0)]
        sector = 1
        finish = False
        for i in range(1, self.board_size):
            for j in range(1, i + 1):
                top_element = reversed_snake[len(reversed_snake) - 1]
                reversed_snake.append((top_element[0] + sector, top_element[1]))
                if len(reversed_snake) >= length:
                    finish = True
                    break
            if finish:
                break

            for j in range(1, i + 1):
                top_element = reversed_snake[len(reversed_snake) - 1]
                reversed_snake.append((top_element[0], top_element[1] + sector))
                if len(reversed_snake) >= length:
                    finish = True
                    break
            if finish:
                break
            sector *= -1
        reversed_snake.reverse()
        result_snake = reversed_snake  # rename the reversed snake

        for i in range(len(result_snake)):
            result_snake[i] = (result_snake[i][0] + self.board_size // 2, result_snake[i][1] + self.board_size // 2)  # shift the snake to the middle
        return result_snake
    
    def _get_init_snake_ground(self, length):
        reversed_snake = [(0, 0)]
        sector = 1
        finish = False
        for i in range(1, self.board_size):
            for j in range(1, i + 1):
                top_element = reversed_snake[len(reversed_snake) - 1]
                reversed_snake.append((top_element[0] + sector, top_element[1]))
                if len(reversed_snake) >= length:
                    finish = True
                    break
            if finish:
                break

            for j in range(1, i + 1):
                top_element = reversed_snake[len(reversed_snake) - 1]
                reversed_snake.append((top_element[0], top_element[1] - sector))
                if len(reversed_snake) >= length:
                    finish = True
                    break
            if finish:
                break
            sector *= -1
        reversed_snake.reverse()
        result_snake = reversed_snake  # rename the reversed snake

        for i in range(len(result_snake)):
            result_snake[i] = (result_snake[i][0] + self.board_size // 2, result_snake[i][1] + self.board_size // 2)  # shift the snake to the middle
        return result_snake
    
    def draw_score(self):
        score_text = self.font.render(f"Length: {len(self.snake)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.border_size, self.height + 2 * self.border_size))
    
    def draw_welcome_screen(self):
        title_text = self.font.render("SNAKE GAME", True, (255, 255, 255))
        start_button_text = "START"

        self.screen.fill((0, 0, 0))
        self.screen.blit(title_text, (self.display_width // 2 - title_text.get_width() // 2, self.display_height // 4))
        self.draw_button_text(start_button_text, (self.display_width // 2, self.display_height // 2))
        pygame.display.flip()

    def draw_game_over_screen(self):
        game_over_text = self.font.render("GAME OVER", True, (255, 255, 255))
        final_score_text = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        retry_button_text = "RETRY"

        self.screen.fill((0, 0, 0))
        self.screen.blit(game_over_text, (self.display_width // 2 - game_over_text.get_width() // 2, self.display_height // 4))
        self.screen.blit(final_score_text, (self.display_width // 2 - final_score_text.get_width() // 2, self.display_height // 4 + final_score_text.get_height() + 10))
        self.draw_button_text(retry_button_text, (self.display_width // 2, self.display_height // 2))          
        pygame.display.flip()

    def draw_button_text(self, button_text_str, pos, hover_color=(255, 255, 255), normal_color=(100, 100, 100)):
        mouse_pos = pygame.mouse.get_pos()
        button_text = self.font.render(button_text_str, True, normal_color)
        text_rect = button_text.get_rect(center=pos)
        
        if text_rect.collidepoint(mouse_pos):
            colored_text = self.font.render(button_text_str, True, hover_color)
        else:
            colored_text = self.font.render(button_text_str, True, normal_color)
        
        self.screen.blit(colored_text, text_rect)
    
    def draw_countdown(self, number):
        countdown_text = self.font.render(str(number), True, (255, 255, 255))
        self.screen.blit(countdown_text, (self.display_width // 2 - countdown_text.get_width() // 2, self.display_height // 2 - countdown_text.get_height() // 2))
        pygame.display.flip()

    def is_mouse_on_button(self, button_text):
        mouse_pos = pygame.mouse.get_pos()
        text_rect = button_text.get_rect(
            center=(
                self.display_width // 2,
                self.display_height // 2,
            )
        )
        return text_rect.collidepoint(mouse_pos)

    def render(self):
        self.screen.fill((0, 0, 0))

        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255), (self.border_size - 2, self.border_size - 2, self.width + 4, self.height + 4), 2)

        # Draw snake
        self.draw_snake()
        
        # Draw food
        if len(self.snake) < self.grid_size: # If the snake occupies the entire board, don't draw food.
            r, c = self.food
            pygame.draw.rect(self.screen, (255, 0, 0), (c * self.cell_size + self.border_size, r * self.cell_size + self.border_size, self.cell_size, self.cell_size))

        # Draw score
        self.draw_score()

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def draw_snake(self):
        # Draw the head
        head_r, head_c = self.snake[0]
        head_x = head_c * self.cell_size + self.border_size
        head_y = head_r * self.cell_size + self.border_size

        # Draw the head (Blue)
        pygame.draw.polygon(self.screen, (100, 100, 255), [
            (head_x + self.cell_size // 2, head_y),
            (head_x + self.cell_size, head_y + self.cell_size // 2),
            (head_x + self.cell_size // 2, head_y + self.cell_size),
            (head_x, head_y + self.cell_size // 2)
        ])

        eye_size = 3
        eye_offset = self.cell_size // 4
        pygame.draw.circle(self.screen, (255, 255, 255), (head_x + eye_offset, head_y + eye_offset), eye_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (head_x + self.cell_size - eye_offset, head_y + eye_offset), eye_size)

        # Draw the body (color gradient)
        color_list = np.linspace(255, 100, len(self.snake), dtype=np.uint8)
        i = 1
        for r, c in self.snake[1:]:
            body_x = c * self.cell_size + self.border_size
            body_y = r * self.cell_size + self.border_size
            body_width = self.cell_size
            body_height = self.cell_size
            body_radius = 5
            pygame.draw.rect(self.screen, (0, color_list[i], 0),
                            (body_x, body_y, body_width, body_height), border_radius=body_radius)
            i += 1
        pygame.draw.rect(self.screen, (0, 100, 255),
                            (body_x, body_y, body_width, body_height), border_radius=body_radius)
        

if __name__ == "__main__":

    seed = random.randint(0, 1e9)
    game = SnakeGame(seed=seed, length=3, is_grow=True, silent_mode=False)
    pygame.init()
    game.screen = pygame.display.set_mode((game.display_width, game.display_height))
    pygame.display.set_caption("Snake Game")
    game.font = pygame.font.Font(None, 36)
    

    game_state = "welcome"

    # Two hidden button for start and retry click detection
    start_button = game.font.render("START", True, (0, 0, 0))
    retry_button = game.font.render("RETRY", True, (0, 0, 0))

    update_interval = 0.15
    start_time = time.time()
    action = -1

    while True:
        
        for event in pygame.event.get():

            if game_state == "running":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 3
                    elif event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 2

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if game_state == "welcome" and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(start_button):
                    for i in range(3, 0, -1):
                        game.screen.fill((0, 0, 0))
                        game.draw_countdown(i)
                        # game.sound_eat.play()
                        pygame.time.wait(1000)
                    action = -1  # Reset action variable when starting a new game
                    game_state = "running"

            if game_state == "game_over" and event.type == pygame.MOUSEBUTTONDOWN:
                if game.is_mouse_on_button(retry_button):
                    for i in range(3, 0, -1):
                        game.screen.fill((0, 0, 0))
                        game.draw_countdown(i)
                        # game.sound_eat.play()
                        pygame.time.wait(1000)
                    game.reset()
                    action = -1  # Reset action variable when starting a new game
                    game_state = "running"
        
        if game_state == "welcome":
            game.draw_welcome_screen()

        if game_state == "game_over":
            game.draw_game_over_screen()

        if game_state == "running":
            if time.time() - start_time >= update_interval:
                done, _ = game.step(action)
                game.render()
                start_time = time.time()

                if done:
                    game_state = "game_over"
        
        pygame.time.wait(1)
