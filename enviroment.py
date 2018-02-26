import time
import random
import queue
import numpy as np
import matplotlib.pyplot as plt
import skimage.io, skimage.feature, skimage.transform, skimage.draw, skimage.morphology, skimage.color, skimage.filters


class MazeBase:
    def __init__(self):
        self._maze_original = None

    def is_not_used(self):
        pass

    def position_is_valid(self, maze, pt, wall_threshold):
        self.is_not_used()
        #
        x, y = pt
        h, w = maze.shape
        #
        if 0 < x < (w - 1) and 0 < y < (h - 1):
            # 3x3 elements (all less than wall_threshold)
            if np.sum(np.less(maze[y - 1:y + 2, x - 1:x + 2], wall_threshold)) == 9:
                return True
        return False

    def make_maze(self, size=(61, 81)):
        self.is_not_used()
        #
        h, w = size
        wall, route = 255, 0
        maze = np.ones([h, w], dtype=np.uint8) * wall
        visited = np.zeros_like(maze, dtype=np.bool)
        edge = np.zeros([h, w, 2], dtype=np.uint32)
        #
        schedule = set()
        x, y = 1, 1
        while True:
            maze[y, x] = route
            visited[y, x] = True
            neighbors = [(x, y - 2), (x, y + 2), (x - 2, y), (x + 2, y)]
            for xx, yy in neighbors:
                if 0 <= xx < w and 0 <= yy < h:
                    if visited[yy, xx]:
                        continue
                    schedule.add((xx, yy))
                    edge[yy, xx] = ((xx + x) // 2, (yy + y) // 2)
            if len(schedule):
                [(x, y)] = random.sample(schedule, 1)
                schedule.remove((x, y))
                xx, yy = edge[y, x]
                maze[yy, xx] = route
            else:
                break
        #
        return maze

    def dynamic_programming(self, maze, src, dst, wall_threshold):
        self.is_not_used()
        #
        h, w = maze.shape
        visited = np.zeros_like(maze, dtype=np.bool)
        trace = np.zeros([h, w, 2], dtype=np.uint32)
        schedule = queue.Queue()
        #
        schedule.put(src)
        while True:
            x, y = schedule.get()
            if visited[y, x]:  # `schedule` may include repeated items
                continue
            visited[y, x] = True
            if (x, y) == dst:
                break
            else:
                for xx, yy in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
                    if 0 <= xx < w and 0 <= yy < h:
                        if visited[yy, xx]:
                            continue
                        if maze[yy, xx] >= wall_threshold:
                            continue
                        schedule.put((xx, yy))
                        trace[yy, xx] = x, y
        path = []
        xx, yy = dst
        while True:
            path.append((xx, yy))
            x, y = trace[yy, xx]
            if src == (x, y):
                break
            xx, yy = x, y
        path.reverse()
        return np.array(path, dtype=np.int)

    def slide_window(self, maze, pt, window_size):
        self.is_not_used()
        #
        wall, route = np.max(maze), 0
        h_maze, w_maze = maze.shape
        #
        x, y = pt
        h_slide, w_slide = window_size

        # expected rect (0, 0) -> (32, 32) not (31, 31)
        top, bottom, left, right = y - h_slide // 2, y + h_slide // 2, x - w_slide // 2, x + w_slide // 2
        # valid rect
        v_top, v_bottom, v_left, v_right = max(top, 0), max(min(bottom, h_maze - 1), 0), \
                                           max(left, 0), max(min(right, w_maze - 1), 0)
        # generate slide window
        sw = np.ones([h_slide, w_slide], dtype=np.uint8) * wall
        if v_top != v_bottom and v_left != v_right:
            sw[v_top - top:h_slide - bottom + v_bottom, v_left - left:w_slide - right + v_right] = \
                maze[v_top:v_bottom, v_left:v_right]
        #
        return sw, v_top, v_bottom, v_left, v_right

    def rotate_window(self, maze, pt, yaw, window_size):
        h, w = window_size
        assert h == w
        radius = h // 2
        #
        sw, top, bottom, left, right = self.slide_window(maze, pt, window_size)
        # rotation
        rw = skimage.transform.rotate(sw, np.rad2deg(yaw))
        # circle view
        rr, cc = skimage.draw.circle(radius - 1, radius - 1, radius)
        cv = np.ones_like(rw)
        cv[rr, cc] = rw[rr, cc]
        #
        return cv, top, bottom, left, right


class SmartNpc:
    def __init__(self, n, nearby_distance):
        self._maze = None
        #
        self._n = n
        self._nearby_distance = nearby_distance
        self._npc = None
        self._direction = None

    def position_is_valid(self, maze, pt, wall_threshold):
        raise NotImplementedError

    def reset_npc(self, positions):
        npc = []
        while True:
            idx = np.arange(0, len(positions))
            np.random.shuffle(idx)
            pts = positions[idx[:self._n]]
            # pts = random.sample(positions, self._n)
            for pt in pts:
                pt = (pt[0] + random.randint(-self._nearby_distance, self._nearby_distance),
                      pt[1] + random.randint(-self._nearby_distance, self._nearby_distance))
                if self.position_is_valid(self._maze, pt, wall_threshold=0.2):
                    npc.append(pt)
            if len(npc) >= self._n:
                break
        self._npc = np.array(npc[:self._n], dtype=np.int)
        self._direction = np.array([random.choice([(0, -1), (0, 1),
                                                   (-1, 0), (1, 0)]) for _ in range(self._n)], dtype=np.int)

    def update_npc(self):
        npc = np.add(self._npc, self._direction)
        for idx, pt in enumerate(npc):
            # hit wall check
            if self.position_is_valid(self._maze, pt, wall_threshold=0.2):
                # probability to change direction
                if random.random() > 0.75:
                    self._direction[idx] = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
            else:
                self._direction[idx] = np.negative(self._direction[idx])
                npc[idx] = self._npc[idx] + self._direction[idx]
        self._npc = npc


class GridEnv(MazeBase, SmartNpc):
    ACTION_AHEAD, ACTION_BACK, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
    FOOT_PRINT = [(0, -3), (0, 3), (-3, 0), (3, 0)]

    def __init__(self, observation_size=(64, 64), allow_control=False):
        MazeBase.__init__(self)
        SmartNpc.__init__(self, n=25, nearby_distance=25)

        # np.float32 array of data for process
        self._maze = None
        self._maze_with_npc = None
        self._path = None
        self._path_idx = 0

        # for visualization
        self._npc = None
        self._img = None
        self._slide_window = None
        self._circle_view = None
        self._top, self._bottom, self._left, self._right = None, None, None, None

        # init matplotlib
        plt.close('all')
        plt.ioff()
        self._fig = plt.figure(figsize=(10, 5))
        if allow_control:
            self._fig.canvas.mpl_connect('key_press_event', self._handler_keydown)
        self._ax1, self._ax2, self._ax3 = plt.subplot(121), plt.subplot(222), plt.subplot(224)
        plt.tight_layout()

        # variable for training
        self._observation_size = observation_size
        self._position = None
        self._views_history = None

    def _update_maze(self):
        self._maze_with_npc = self._maze.copy()
        for pt in self._npc:
            x, y = pt
            rr, cc = skimage.draw.circle(y, x, 2)
            self._maze_with_npc[rr, cc] = 1.0

    def _update_img(self):
        self._img = skimage.color.gray2rgb(np.subtract(1.0, self._maze))
        for pt in self._npc:
            x, y = pt
            rr, cc = skimage.draw.circle(y, x, 2)
            self._img[rr, cc] = 1.0, 0.0, 0.0
        rr, cc = skimage.draw.line(self._top, self._left, self._top, self._right)
        self._img[rr, cc] = 1.0, 1.0, 0.0
        rr, cc = skimage.draw.line(self._top, self._right, self._bottom, self._right)
        self._img[rr, cc] = 1.0, 1.0, 0.0
        rr, cc = skimage.draw.line(self._bottom, self._right, self._bottom, self._left)
        self._img[rr, cc] = 1.0, 1.0, 0.0
        rr, cc = skimage.draw.line(self._bottom, self._left, self._top, self._left)
        self._img[rr, cc] = 1.0, 1.0, 0.0

        rr, cc = self._path[:, 1], self._path[:, 0]
        self._img[rr, cc] = 0.0, 1.0, 0.0

    def reset(self):
        # generate original binary maze
        self._maze_original = self.make_maze((21, 21))

        # generate np.float32 maze for later use
        maze = skimage.transform.resize(self._maze_original, (240, 240), mode='edge')
        maze = maze > skimage.filters.threshold_mean(maze)
        self._maze = np.array(
            # skimage.morphology.erosion(maze, selem=skimage.morphology.square(7)),
            skimage.filters.gaussian(skimage.morphology.erosion(maze, selem=skimage.morphology.square(7))),
            dtype=np.float32
        )

        # generate dilated maze for path planing
        maze_for_dp = skimage.morphology.dilation(maze, skimage.morphology.square(7))

        # random choice start and end point
        while True:
            h, w = maze_for_dp.shape
            src, dst = (random.randrange(w), random.randrange(h)), (random.randrange(w), random.randrange(h))

            # manhattan distance
            if np.abs(src[0] - dst[0]) + np.abs(src[1] - dst[1]) < 100:
                continue

            # this point and nearby point all available
            if self.position_is_valid(maze_for_dp, src, wall_threshold=True) and \
                    self.position_is_valid(maze_for_dp, dst, wall_threshold=True):
                break

        # dijkstra
        self._path = self.dynamic_programming(maze_for_dp, src, dst, wall_threshold=0.5)
        self._path_idx = 0
        #
        self.reset_npc(self._path)
        self._update_maze()
        while True:
            x, y = self._path[0]
            x, y = x + random.randint(-5, 5), y + random.randint(-5, 5)
            if self.position_is_valid(self._maze_with_npc, (x, y), wall_threshold=0.2):
                self._position = (x, y)
                break

        state_init, _, _, _ = self.step(action=None)
        return state_init

    def render(self, pause=1.0):
        self._update_img()
        self._ax1.imshow(self._img)
        if self._slide_window is not None:
            self._ax2.imshow(self._slide_window, cmap='Greys')
        if self._circle_view is not None:
            self._ax3.imshow(self._circle_view, cmap='Greys')
        plt.pause(pause)

    def _handler_keydown(self, event):
        if event.key == 'up':
            action = self.ACTION_AHEAD
        elif event.key == 'down':
            action = self.ACTION_BACK
        elif event.key == 'left':
            action = self.ACTION_LEFT
        elif event.key == 'right':
            action = self.ACTION_RIGHT
        else:
            action = None

        if action is not None:
            self._position = np.add(self._position, [(0, -1), (0, 1), (-1, 0), (1, 0)][action])

    def step_follow_path(self):
        self._position = self._path[self._path_idx]
        self._path_idx += 1
        self.step(action=self.random_action())

    def random_action(self):
        action = np.random.choice([self.ACTION_AHEAD, self.ACTION_BACK, self.ACTION_LEFT, self.ACTION_RIGHT])
        return action

    def step(self, action):
        """
        :param action
        [AHEAD, BACK, LEFT, RIGHT]
        :return state_, reward, done, info
        [cv_(t-4), cv_(t-3), cv_(t-2), cv_(t-1), cv_(t), target_position]
        """
        state_, reward, done = None, 0.0, False
        target_position = self._path[self._path_idx]

        # first, move self
        if action is not None:
            self._position = np.add(self._position, self.FOOT_PRINT[action])

        # second, check collision
        if self.position_is_valid(self._maze_with_npc, self._position, 0.2) is False:
            reward = -1.0
            done = True

        # third, generate current observation after move and env change
        self.update_npc()
        self._update_maze()
        self._slide_window, self._top, self._bottom, self._left, self._right \
            = self.slide_window(self._maze_with_npc, self._position, self._observation_size)

        # forth, check collision again
        if self.position_is_valid(self._maze_with_npc, self._position, 0.2) is False:
            reward = -1.0
            done = True

        # fifth, check reward
        if not done and np.linalg.norm(self._position - target_position) <= 5.0:
            self._path_idx += 1
            reward = 1.0 * self._path_idx
            if self._path_idx == len(self._path):
                done = True

        # views in history positions
        if self._views_history is None:
            self._views_history = np.zeros([5, *self._observation_size], dtype=np.float32)
            self._views_history[:] = self._slide_window
        else:
            self._views_history[:4] = self._views_history[1:]
            self._views_history[4] = self._slide_window

        # [view_(t-4), view_(t-3), view_(t-2), view_(t-1), view_(t), target_position]
        state_ = np.zeros([6, *self._observation_size], dtype=np.float32)
        state_[:-1, :, :] = self._views_history.copy()

        # generate target_position in state_[4]
        # target_position is a one-hot data, which to indicate target in local map
        target_offset = np.subtract(target_position, self._position)
        if abs(target_offset[0]) < 31 and abs(target_offset[1]) < 31:
            h, w = self._observation_size
            tx, ty = np.add(target_offset, [w // 2, h // 2])
            state_[-1, ty - 1:ty + 2, tx - 1:tx + 2] = 1.0
        else:
            state_[-1, :, :] = 0.0
            reward = -1.0
            done = True
        self._ax3.imshow(state_[-1])

        return state_, reward, done, None


def test_env():
    env = GridEnv(allow_control=True)
    env.reset()
    before = time.time()
    for _ in range(1000):
        action = [0, 1][random.randrange(2)]
        state_, reward, done, info = env.step(action)
        print('reward: {0}, done: {1}'.format(reward, done))
        env.render()
    after = time.time()
    print(1000 / (after - before))


def test_env_2():
    env = GridEnv(allow_control=True)
    env.reset()
    for _ in range(1000):
        env.step_follow_path()
        env.render()


if __name__ == '__main__':
    test_env_2()
