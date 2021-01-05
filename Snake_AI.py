import pygame
import neat
import random
import os
from neat.math_util import softmax
import math
import numpy as np

pygame.init()
WIN_WIDTH = 700
WIN_HEIGHT = 500


blue = (0, 0, 255)
red = (255, 0, 0)
black = (0, 0, 0)

snake_list = []


class Snake:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tick_count = 0
        self.length_of_snake = 1
        self.update_x = 0
        self.update_y = 0
        snake_list.clear()

    def draw(self, window, snake_list):
        # draw the snake based on how many fruit it has ate
        for x in snake_list:
            pygame.draw.rect(window, blue, [x[0], x[1], 10, 10])

    def move(self, window, direction):

        if direction == "LEFT":
            self.update_x = -10
            self.update_y = 0
        elif direction == 'RIGHT':
            self.update_x = 10
            self.update_y = 0
        elif direction == 'DOWN':
            self.update_x = 0
            self.update_y = 10
        elif direction == "UP":
            self.update_x = 0
            self.update_y = -10

        self.x += self.update_x
        self.y += self.update_y

        snake_head = []
        snake_head.append(self.x)
        snake_head.append(self.y)
        snake_list.append(snake_head)

        if len(snake_list) > self.length_of_snake:
            del snake_list[0]

        # collision with body, doing nothing for now
        for x in snake_list[:-1]:
            if x == snake_head:
                pass

        self.draw(window, snake_list)

    def get_coordinates(self):
        x = self.x
        y = self.y
        return x, y

    # collision with fruit
    def collide(self, fruits):
        x, y = fruits.get_coordinates()
        if x == self.x and y == self.y:
            self.length_of_snake += 1
            return True

        return False

    def wall_collide(self):
        if self.x < 0 or self.x > WIN_WIDTH - 10:
            return True
        if self.y < 0 or self.y > WIN_HEIGHT - 10:
            return True

    def within_radius_of_food(self, food_pos):
        return math.sqrt((self.x - food_pos[0]) ** 2 + (self.y - food_pos[1]) ** 2)

    def get_distances(self, coordinates):
        pos = []
        directions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
        distances = []
        food_pos = coordinates

        for direction in directions:
            pos.clear()
            pos.append(self.x)
            pos.append(self.y)

            if direction == 'NORTH':
                pos[1] -= 10
                if pos[1] >= WIN_HEIGHT / 2:
                    distances.append(WIN_HEIGHT - pos[1])
                elif pos[1] < WIN_HEIGHT / 2:
                    distances.append(pos[1])
            elif direction == 'SOUTH':
                pos[1] += 10
                if pos[1] >= WIN_HEIGHT / 2:
                    distances.append(WIN_HEIGHT - pos[1])
                elif pos[1] < WIN_HEIGHT / 2:
                    distances.append(pos[1])
            elif direction == 'WEST':
                pos[0] -= 10
                if pos[0] >= WIN_WIDTH / 2:
                    distances.append(WIN_WIDTH - pos[0])
                elif pos[0] < WIN_WIDTH / 2:
                    distances.append(pos[0])
            elif direction == 'EAST':
                pos[0] += 10
                if pos[0] >= WIN_WIDTH / 2:
                    distances.append(WIN_WIDTH - pos[0])
                elif pos[0] < WIN_WIDTH / 2:
                    distances.append(pos[0])
           
            distances.append(math.sqrt((pos[0] - food_pos[0]) ** 2 + (pos[1] - food_pos[1]) ** 2))
            # distances.append(math.sqrt((pos[0]-snake_list[0][0]) ** 2 + (pos[1] - snake_list[0][1]) ** 2))
        # appends 3 distances for each direction of the snake.
        # distance to the wall, food, and tail of the snake
        return distances


class Fruit:
    def __init__(self):
        self.x = round(random.randrange(0, WIN_WIDTH - 10) / 10.0) * 10.0
        # multiply by 10 to get a multiple of 10
        self.y = round(random.randrange(0, WIN_HEIGHT - 10) / 10.0) * 10.0

    def draw(self, window):
        pygame.draw.rect(window, red, [self.x, self.y, 10, 10])

    def get_coordinates(self):
        x = self.x
        y = self.y
        return x, y


def draw_window(window, fruits, snake, direction):
    # window.blit(IMG_BIG, (0, 0))

    window.fill(black)

    fruits.draw(window)

    # for snakes in snake:

    if direction == 'UP':
        snake.move(window, direction)
    elif direction == 'DOWN':
        snake.move(window, direction)
    elif direction == 'LEFT':
        snake.move(window, direction)
    elif direction == 'RIGHT':
        snake.move(window, direction)

    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    snake = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snake.append(Snake(100, 100))
        g.fitness = 0
        ge.append(g)

    clock = pygame.time.Clock()
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    fruits = Fruit()

    for x, g in enumerate(ge):
        run = True
        # snake = Snake(100, 100)
        direction = "RIGHT"
        change_to = direction
        snake[x].move(window, direction)
        moves = 0
        max_moves = 500
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
            # print(snakes.get_coordinates())

           # for x, snake in enumerate(snakes):
               # snake.move(window)
               # ge[x].fitness += 0.1
            fruit_coordinates = fruits.get_coordinates()
            inputs = snake[x].get_distances(fruit_coordinates)

            net_output = nets[x].activate(inputs)
            index = np.argmax(net_output)
            # softmax_result = softmax(net_output)
            # index = softmax_result.index(max(softmax_result))

            if index == 0:
                change_to = 'UP'
            elif index == 1:
                change_to = 'DOWN'
            elif index == 2:
                change_to = 'LEFT'
            elif index == 3:
                change_to = 'RIGHT'

            # Making sure the snake cannot move in the opposite direction instantaneously
            if change_to == 'UP' and direction != 'DOWN':
                direction = 'UP'
            if change_to == 'DOWN' and direction != 'UP':
                direction = 'DOWN'
            if change_to == 'LEFT' and direction != 'RIGHT':
                direction = 'LEFT'
            if change_to == 'RIGHT' and direction != 'LEFT':
                direction = 'RIGHT'


           # if not snake.wall_collide():
              #  g.fitness += 0.
            snake_coordinates = snake[x].get_coordinates()
            # encourage snake to go towards food if it in the same x-y plane
            if snake_coordinates[0] == fruit_coordinates[0]:
                g.fitness += 5
            if snake_coordinates[1] == fruit_coordinates[1]:
                g.fitness += 5
            if snake[x].collide(fruits):
                fruits = Fruit()
                # reset moves upon getting fruit
                moves = 0
                g.fitness += 15
            # encourage snake to go towards food, increase fitness if it is within a certain radius
            elif snake[x].within_radius_of_food(fruit_coordinates) < 40:
                g.fitness += 5
            elif snake[x].within_radius_of_food(fruit_coordinates) < 60:
                g.fitness += 2
            else:
                g.fitness -= 1
            # print(snake[x].within_radius_of_food(coordinates))
            # if not snake[x].wall_collide():
               # g.fitness += 0.1

            moves += 1

            if run:
                draw_window(window, fruits, snake[x], direction)

            if moves > max_moves:
                g.fitness -= 5
                snake.pop(x)
                nets.pop(x)
                ge.pop(x)
                run = False

            if snake[x].wall_collide():
                g.fitness -= 30
                snake.pop(x)
                nets.pop(x)
                ge.pop(x)
                run = False

def run_game(config_paths):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_paths)

    p = neat.Population(config)
    # print out statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 150 is how many generations
    winner = p.run(main, 200)
    # passes main all of the configs, and genomes


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "Snake_NEAT.txt")
    run_game(config_path)


