#coding: utf-8
import numpy as np
import logging
import logging.config

def regularize_x(x, min_x, max_x, decay):
    while True:
        if x < min_x:
            x = min_x - decay * (x - min_x)
        elif x > max_x:
            x = max_x - decay * (x - max_x)
        else:
            break
    return x
"""
def regularize_x(x):
    if x < Particle.min_x:
        x = Particle.min_x - w * (x - Particle.min_x)
    return x
"""


class Particle(object):
    def __init__(self, best_x, best_score=0, init_v_rate=0.1, min_x=0, max_x=2):
        self.best_x = best_x
        self.x = best_x
        self.v = init_v_rate * np.mean(best_x) * np.random.randn(best_x.shape[0])
        self.best_score = best_score
        self.score = best_score
        self.min_x = min_x
        self.max_x = max_x

    def update_x(self, best_x, c1, c2, w):
        rand1 = np.random.rand(self.v.shape[0])
        rand2 = np.random.rand(self.v.shape[0])
        self.v = w * self.v + c1 * rand1 * (self.best_x - self.x) + c2 * rand2 * (best_x - self.x)
        self.x = self.x + self.v
        self.x = np.vectorize(regularize_x)(self.x, self.min_x, self.max_x, w)
    
    def update_score(self, score):
        self.score = score
        if score > self.best_score:
            self.best_x = self.x
            self.best_score = score

class PSO(object):
    def __init__(self, best_x, score_function, group_size, min_x, max_x):
        self.score_function = score_function
        self.gbest = Particle(best_x, score_function(best_x))
        self.ps = [Particle(best_x, score_function(best_x))]

        x_scale = max_x - min_x
        for i in range(1, group_size):
            x = min_x + x_scale * np.random.rand(best_x.shape[0])
            score = score_function(x)
            p = Particle(x, score)
            if score > self.gbest.score:
                self.gbest = Particle(x, score)
            self.ps.append(p)

    
    def update(self, c1, c2, w, max_iter, patience):
        logger = logging.getLogger(__name__)
        logger.info("init solution: gbest.x: %s; gbest.score: %f" % (self.gbest.x, self.gbest.score))
        cnt = 0
        for i in range(1, max_iter+1):
            logger.info("PSO_iter %d" %(i))
            for p in self.ps:
                p.update_x(self.gbest.best_x, c1, c2, w)
                score = self.score_function(p.x)
                p.update_score(score)
                if score > self.gbest.score:
                    self.gbest = Particle(p.best_x, p.best_score)
                    logger.info("find a new better solution: gbest.x: %s; gbest.score: %f" % (self.gbest.x, self.gbest.score))
                    cnt = 0
            cnt += 1
            if cnt >= patience:
                break
        logger.info("PSO finished! The best solution: gbest.x: %s; gbest.score: %f" % (self.gbest.x, self.gbest.score))

    def get_best_x(self):
        return self.gbest.best_x