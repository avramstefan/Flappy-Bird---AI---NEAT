import pygame
import random
import time
import pipe_random_gen
import neat
import os
import math


######### Preparing the visual elements ############

pygame.init()

HEIGHT = 700
WIDTH = 500
WHITE = (255, 255, 255)

GEN = 0

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Preparing the frames for background_gif // splitting the gif into images / frames
back_img = ["images/back_images/frame_0" + str(i) + "_delay-0.03s.gif" for i in range(10)]
for i in range(10, 90):
    back_img.append("images/back_images/frame_" + str(i) + "_delay-0.03s.gif")
back_frames = []
for name in back_img:
    back_frames.append(pygame.image.load(name))

# loading the pipe body
pipe_up = pygame.image.load(r'images/up-down-pipe-edit.png')
pipe_body_up = pygame.image.load(r'images/up-down-pipe-body-edit.png')
pipe_down = pygame.image.load(r'images/down-up-pipe-edit.png')
pipe_body_down = pygame.image.load(r'images/down-up-pipe-body-edit.png')

# Game over image
game_over = pygame.image.load(r'images/gameover.png')

# Bird image
character = pygame.image.load(r'images/red-bird.png')

font1 = pygame.font.SysFont('freesanbold.ttf', 50) # font_style

########################################


###### Classes ######

class Bird():
    
    acceleration = 1
    falling = False

    # Physics -> v1 = v0 + a * (t1 - t0);
    #               where a = acceleration
    #                     t1 = momentum at time 1, while t0 = momentum at time 0
    #                     v1 = velocity at time 1, while v0 = velocity at time 0
    v0 = 5
    v1 = 5

    def __init__(self, bird_x_pos, bird_y_pos):
        self.bird_x_pos = bird_x_pos
        self.bird_y_pos = bird_y_pos
        
    def update_pos(self):
        self.v1 = self.v0 + self.acceleration * 0.015
        self.bird_y_pos += self.v1
        if self.bird_x_pos < 205:
            self.bird_x_pos += 1
        self.v0 = self.v1

    def jump_update_pos(self):
        
        self.bird_y_pos -= 10
        if self.bird_x_pos < 205:
            self.bird_x_pos += 2
        
    
class Pipe():
    
    pipe_slide_units = 2.5
    
    def __init__(self, x_pos, y_pos, x_pos_1, y_pos_1, x_pos_2, y_pos_2):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_pos_1 = x_pos_1
        self.y_pos_1 = y_pos_1
        self.x_pos_2 = x_pos_2
        self.y_pos_2 = y_pos_2
        
    def update_pos(self):
        self.x_pos -= self.pipe_slide_units
        self.x_pos_1 -= self.pipe_slide_units
        self.x_pos_2 -= self.pipe_slide_units


#######################################
        
def background_draw(window, birds, pipes, is_game_over, frames_idx, score):
    window.fill(WHITE)
    
    # Background 
    window.blit(back_frames[frames_idx % 89], (-100, 0)) # 0...89 frames
    
    # Character
    if not is_game_over:
        for bird in birds:
            window.blit(character, (bird.bird_x_pos, bird.bird_y_pos))
    
    # Pipes
    for i in range(len(pipes)):
        window.blit(pipe_body_up, (pipes[i][0].x_pos_1, pipes[i][0].y_pos_1))
        window.blit(pipe_body_up, (pipes[i][0].x_pos_2, pipes[i][0].y_pos_2))
        window.blit(pipe_up, (pipes[i][0].x_pos, pipes[i][0].y_pos))
        
        window.blit(pipe_body_down, (pipes[i][1].x_pos_1, pipes[i][1].y_pos_1))
        window.blit(pipe_body_down, (pipes[i][1].x_pos_2, pipes[i][1].y_pos_2))
        window.blit(pipe_down, (pipes[i][1].x_pos, pipes[i][1].y_pos))
        
    # Text
    if not is_game_over:
        text1 = font1.render('Score: ' + str(score), True, (10, 255, 255))
        textRect1 = text1.get_rect()
        textRect1.center = (250, 20)
        window.blit(text1, textRect1)
    else:
        for i in range(len(pipes)):
            pipes[i][0].update_pos()
            pipes[i][1].update_pos()
                    
        if pipes[0][0].x_pos <= -115:
            pipes = pipes[1:]
        
        window.blit(game_over, (10, 200))
        time.sleep(0.015)
        
    # Apply changes
    pygame.display.update()
    
def check_score(bird, pipes):
    if len(pipes) and ((bird.bird_x_pos > pipes[0][0].x_pos + 114 and
        bird.bird_x_pos < pipes[0][0].x_pos + 116) or
        (len(pipes) > 2 and bird.bird_x_pos > pipes[1][0].x_pos + 114 and
         bird.bird_x_pos < pipes[1][0].x_pos + 116)):
        return True
    return False
        
def check_if_collision(x, y, pipes):
    for i in range(len(pipes)):
        if y <= 0 or y >= 700 or ((x + 60 >= pipes[i][0].x_pos and x <= pipes[i][0].x_pos + 115) and not
                                  (y >= pipes[i][0].y_pos + 262 and y + 44 <= pipes[i][1].y_pos)):
            return True 
    return False

def distance_from_bird_to_pipe(bird_x, bird_y, pipe_x, pipe_y):
    c1 = abs(bird_x - pipe_x)
    c2 = abs(bird_y - pipe_y)
    return int(math.sqrt(c1 ** 2 + c2 ** 2))

def main(genomes, config):
    pygame.init()
    
    frames_idx = 0
    score = 0

    # X coordinate starting position for pipes
    x_pipe_start = 500
    
    global window, GEN
    GEN += 1
    
    pipes = []
    frames_idx += 1
    
    nets = []
    birds = []
    ge = []

    # Creating birds and adding them to be monitorized by NEAT
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(100,350))
        ge.append(genome)

    background_draw(window, birds, pipes, 0, frames_idx, score)
    next_y_jump = 0
    wait_until_next_jump = [False for _ in range(len(birds))]
    height_of_jump = [0 for _ in range(len(birds))]
    prev_score = score

    """
    Birds' movement.

    Using a tanh function (not sigmoid). Tanh function has values between -1 and 1.

    Therefore, if the output of the function is higher, then the probability
    that the bird should jump increases. 

    The parameteres that are being considered are represented by the bird's
    coordinates and the distances from the bird to the next pipe (obstacle).
    """
    while True and len(birds):

        next_pipe = 0
        for bird in birds:
            if len(pipes) and bird.bird_x_pos >= pipes[0][0].x_pos + 115:
                next_pipe = 1
                break
        
        # For each second in the runtime that the bird survives, a point is added to its fitness.
        for x, bird in enumerate(birds): 
            ge[x].fitness += 0.1

            if wait_until_next_jump[x]:
                if bird.bird_y_pos <= height_of_jump[x]:
                    wait_until_next_jump[x] = False
                else:
                    bird.bird_y_pos -= 5
                    continue

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            output = [0]
            if len(pipes) and not wait_until_next_jump[x]:
                output = nets[birds.index(bird)].activate((bird.bird_y_pos,
                    distance_from_bird_to_pipe(bird.bird_x_pos, bird.bird_y_pos,
                    pipes[next_pipe][0].x_pos, pipes[next_pipe][0].y_pos + 262),
                    distance_from_bird_to_pipe(bird.bird_x_pos, bird.bird_y_pos,
                    pipes[next_pipe][1].x_pos, pipes[next_pipe][1].y_pos - 44)))

            if (len(pipes) == 0) and not wait_until_next_jump[x]:
                bird.bird_x_pos += 2
            # tanh activation function so result will be between in between [-1,1]. over 0.7 means jump
            elif ((len(pipes) == 0) or (len(pipes) and output[0] > 0.7)) and not wait_until_next_jump[x]:
                next_y_jump = bird.bird_y_pos
                wait_until_next_jump[x] = True
                height_of_jump[x] = bird.bird_y_pos - 50
                bird.falling = False
                bird.jump_update_pos()
            # where the bird is falling
            elif bird.bird_x_pos >= 205 and not wait_until_next_jump[x]:

                if not bird.falling:
                    bird.v1 = 5
                    bird.v0 = 5
                    bird.falling = True
                bird.update_pos()


        # Creating pipes
        if birds[0].bird_x_pos >= 205:
            if not len(pipes) or (pipes[0][0].x_pos < 100 and len(pipes) == 1):
                if len(pipes) == 1:
                    score += 1
                random_pos = pipe_random_gen.random_pos_up()
                up_pipe_xx = Pipe(x_pipe_start, random_pos, x_pipe_start,
                                  random_pos - 180, x_pipe_start, random_pos - 360)
                down_pipe_xx = Pipe(x_pipe_start, random_pos + 430, x_pipe_start,
                                     random_pos + 680, x_pipe_start, random_pos + 830)
                pipes.append([up_pipe_xx, down_pipe_xx])
            else:
                for i in range(len(pipes)):
                    pipes[i][0].update_pos()
                    pipes[i][1].update_pos()
                    
                if pipes[0][0].x_pos <= -115:
                    pipes = pipes[1:]
                    
        # check_collisions
        for x, bird in enumerate(birds):
            if check_if_collision(bird.bird_x_pos, bird.bird_y_pos, pipes):
                ge[birds.index(bird)].fitness -= 1
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        if score > prev_score:
            for x, bird in enumerate(birds): 
                ge[x].fitness += 0.5
            
        background_draw(window, birds, pipes, 0, frames_idx, score)
        frames_idx += 1
        time.sleep(0.013)        
        
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Run 50 times
    winner = p.run(main, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
        
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
        