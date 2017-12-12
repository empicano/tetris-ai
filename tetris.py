import curses
import math
import random
import time
import json


ROWS = 22  # official board: 22 rows
COLS = 10  # official board: 10 columns
LETTERS = 'IJLOSZT'

# binary representation of tetromino shapes
SHAPES = [
    [0, 15, 0, 0],  # I
    [4, 7, 0],  # J
    [1, 7, 0],  # L
    [3, 3],  # O
    [3, 6, 0],  # S
    [6, 3, 0],  # Z
    [2, 7, 0]  # T
]


class Tetromino:
    """Represents one tetromino"""

    def __init__(self, letter):
        """Sets starting position and maps letter to matrix shape"""
        self.pos = (4, 0) if letter == 'O' else (3, 0)  # format: (x, y)
        self.letter = letter
        self.color = LETTERS.index(self.letter) + 1
        self.shape = SHAPES[self.color - 1]
        self.length = len(self.shape)

    def step(self, board, direction):
        """Moves tetromino one field to direction, if possible; Returns True if successful, False if not"""
        mv = (1, 0) if direction == 'RIGHT' else (-1, 0) if direction == 'LEFT' else (0, 1)
        status = True
        self.draw(board, colored=False)
        self.pos = (self.pos[0] + mv[0], self.pos[1] + mv[1])
        if not self.pos_valid(board):
            self.pos = (self.pos[0] - mv[0], self.pos[1] - mv[1])
            status = False
        self.draw(board)
        return status

    def rotate(self, board):
        """Rotates tetromino 90 degrees clockwise if possible"""
        self.draw(board, colored=False)
        self.shape = self.bit_matrix_turn(self.shape)
        if not self.pos_valid(board):
            self.shape = self.bit_matrix_turn(self.shape, clockwise=False)
        self.draw(board)

    def draw(self, board, colored=True, bit=False):
        """Adds tetromino to the board"""
        for y in range(self.length):
            for x in range(self.length):
                if self.shape[y] >> (self.length - 1 - x) & 1:
                    if bit:
                        board[self.pos[1] + y] += (1 if colored else -1) * (1 << (COLS - 1 - self.pos[0] - x))
                    else:
                        board[self.pos[1] + y][self.pos[0] + x] = self.color if colored else 0

    def pos_valid(self, board, bit=False):
        """Checks if current position is allowed; Returns True if yes, False if not"""
        for y in range(self.length):
            for x in range(self.length):
                if self.shape[y] >> (self.length - 1 - x) & 1:
                    if not 0 <= (self.pos[1] + y) < ROWS or not 0 <= (self.pos[0] + x) < COLS:
                        return False
                    if bit:
                        if board[self.pos[1] + y] >> (COLS - 1 - self.pos[0] - x) & 1:
                            return False
                    elif board[self.pos[1] + y][self.pos[0] + x]:
                        return False
        return True

    @staticmethod
    def bit_matrix_turn(matrix, clockwise=True):
        """Returns given n*n bit-matrix rotated by 90 degrees"""
        res = []
        n = len(matrix)
        for p in (range(n - 1, -1, -1) if clockwise else range(n)):
            row = 0
            for i in range(n):
                row += (matrix[i] >> p & 1) << (i if clockwise else n - i - 1)
            res.append(row)
        return res


class Screen:
    """Represents the terminal screen"""

    BORDER_LEFT = 3
    BORDER_TOP = 3
    MIN_LEVEL = 1
    MAX_LEVEL = 10

    def __init__(self, vis):
        """Initiates curses; Prepares terminal; Sets color pairs; Builds board"""
        self.board = None
        self.score = 0
        self.level = 0

        if vis:
            # prepare terminal
            self.sdtscr = curses.initscr()
            self.sdtscr.nodelay(True)
            self.sdtscr.keypad(True)
            curses.start_color()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(False)

            # set tetromino colors
            for color in range(1, 8):
                curses.init_pair(color, 0, color)

            # build background borders
            self._draw_matrix(self.BORDER_LEFT, self.BORDER_TOP, [['| '] for _ in range(ROWS - 2)])
            self._draw_matrix(self.BORDER_LEFT + COLS + 1, self.BORDER_TOP, [[' |'] for _ in range(ROWS - 2)])

        # build blank board
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        if vis: self.update_board()

    def _draw_text(self, x, y, text='  ', color=0):
        """Draws text colored with specified color-pair to position (x, y)"""
        self.sdtscr.addstr(y, x, text, curses.color_pair(color))

    def _draw_matrix(self, x, y, board):
        """Draws matrix at position (x, y) to the terminal"""
        for row in range(len(board)):
            for col in range(len(board[0])):
                if isinstance(board[row][col], int):
                    self._draw_text((col + x) * 2, row + y, color=board[row][col])
                else:
                    self._draw_text((col + x) * 2, row + y, text=board[row][col])

    def remove_rows(self):
        """Removes every complete horizontal row"""
        for i in range(ROWS):
            if all(val != 0 for val in self.board[i]):
                del self.board[i]
                self.board = [[0 for _ in range(COLS)]] + self.board
                self.score += 1

    def update_next_tetro(self, tetro):
        """Updates next up tetromino display"""
        pos = (self.BORDER_LEFT + COLS + 4, self.BORDER_TOP + 4)
        self._draw_matrix(*pos, [[0 for _ in range(3 - len(tetro.shape))] + \
            [tetro.color * (row >> i & 1) for i in range(tetro.length - 1, -1, -1)] + [0]
            for row in tetro.shape])

    def update_stats(self):
        """Updates level and score displays"""
        pos = ((self.BORDER_LEFT + COLS + 4) * 2, self.BORDER_TOP)
        if self.level < self.MAX_LEVEL:
            self.level = max(self.score // 10 + 1, self.MIN_LEVEL)
            self._draw_text(pos[0], pos[1], text='LEVEL:' + '{:8}'.format(self.level))
        self._draw_text(pos[0], pos[1] + 1, text='SCORE:' + '{:8}'.format(self.score))

    def update_message(self, text):
        """Prints message to the screen; max 20 characters in length"""
        assert len(text) <= 20
        # overwrite last message by appending whitespaces
        self._draw_text((self.BORDER_LEFT + COLS + 4) * 2, self.BORDER_TOP + ROWS - 3, text + ' ' * (20 - len(text)))
        self.sdtscr.refresh()

    def update_board(self):
        """Updates terminal; The first two rows of board are hidden"""
        self._draw_matrix(self.BORDER_LEFT + 1, self.BORDER_TOP, self.board[2:])
        self.sdtscr.refresh()

    def bit_rep(self):
        """Returns binary representation of current board"""
        res = []
        for y in range(ROWS):
            row = 0
            for x in range(COLS):
                if self.board[y][x]:
                    row += (1 << (COLS - 1 - x))
            res.append(row)
        return res

    def cleanup(self):
        """Returns terminal to its original state"""
        self.sdtscr.clear()
        self.sdtscr.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.curs_set(True)
        curses.endwin()


class AI:
    """Provides methods for heuristic computing of the best move for the current tetris board"""

    @staticmethod
    def get_best_move(board, dna, tetro, follow):
        """Chooses the best tetro move by also further evaluating their combination with the following tetromino"""
        sv = tetro.pos[:]
        best = AI._get_best_pos(board, dna, tetro, slide=3)  # get the 3 best moves
        final = None

        for m in range(len(best)):
            # set tetromino; remove full rows
            tmp = board[:]
            tetro.pos = best[m][1]
            for r in range(best[m][2]):
                tetro.shape = Tetromino.bit_matrix_turn(tetro.shape)
            tetro.draw(tmp, bit=True)
            for i in range(ROWS):
                if tmp[i] == 1023:
                    del tmp[i]
                    tmp.insert(0, 0)
            # evaluate board with following tetromino
            future = AI._get_best_pos(tmp, dna, follow, slide=1)[0]
            if final is None or future[0] > final[0]:
                final = [future[0], m]
            for r in range(best[m][2]):
                tetro.shape = Tetromino.bit_matrix_turn(tetro.shape, clockwise=False)

        tetro.pos = sv
        return best[final[1]][1], best[final[1]][2]

    @staticmethod
    def _get_best_pos(board, dna, tetro, slide):
        """Tries every move with given tetro, evaluates the resulting boards and returns the best ones"""
        best = []
        sv = tetro.pos[:]
        tmp = board[:]

        # check different rotations of shape
        for r in range(1 if tetro.letter == 'O' else 2 if tetro.letter in 'SZI' else 4):
            for x in range(-2, COLS - 1):
                for y in range(ROWS):
                    tetro.pos = (x, y)
                    if not tetro.pos_valid(tmp, bit=True):
                        # stop when tetro can't fall freely to the position anymore
                        break

                if tetro.pos[1] > 0:
                    # set tetromino; remove full rows; evaluate
                    tetro.pos = (tetro.pos[0], tetro.pos[1] - 1)
                    tetro.draw(tmp, bit=True)
                    changed = False
                    for i in range(ROWS):
                        if tmp[i] == 1023:
                            del tmp[i]
                            tmp.insert(0, 0)
                            changed = True
                    res = AI._evaluate(tmp, dna)
                    # make a new copy of board if rows were removed, else simply delete last tetromino
                    if changed: tmp = board[:]
                    else: tetro.draw(tmp, colored=False, bit=True)
                    # save the best of all possible positions
                    for i in range(slide):
                        if len(best) == i or res > best[i][0]:
                            best.insert(i, [res, tetro.pos[:], r])
                            if len(best) > slide: del best[slide]
                            break

            tetro.shape = Tetromino.bit_matrix_turn(tetro.shape)

        # reset tetro
        if tetro.letter in 'SZI':
            tetro.shape = Tetromino.bit_matrix_turn(Tetromino.bit_matrix_turn(tetro.shape))
        tetro.pos = sv
        return best

    @staticmethod
    def _evaluate(board, dna):
        """Evaluates board considering different factors to compute a score (the higher the better)"""
        # count gaps (empty spaces under blocks)
        gaps = 0
        bumps = []
        for x in range(COLS):
            blocked = False
            for y in range(ROWS):
                if blocked and not (board[y] >> x & 1):
                    gaps += 1
                elif not blocked:
                    if board[y] >> x & 1:
                        bumps.append(ROWS - y)
                        blocked = True
                    elif y == ROWS - 1:  # add zero to bumps list when column is empty
                        bumps.append(0)
        # get added height of all columns
        height = sum(bumps)
        # compute bumpiness (absolute difference of adjacent columns)
        bumpiness = 0
        for i in range(len(bumps) - 1):
            bumpiness += abs(bumps[i] - bumps[i + 1])
        # return score
        return (dna[0] * gaps) + (dna[1] * height) + (dna[2] * bumpiness)


class Genetics:
    """Provides methods to improve ai performance"""

    @staticmethod
    def _normalize_vector(vec):
        """Returns normalized version of given vector"""
        length = 0
        for e in vec:
            length += math.pow(e, 2)
        length = math.sqrt(length)
        return [e / length for e in vec]

    @staticmethod
    def create_population(count=100, weights=3):
        """Creates random population and saves it to text file"""
        population = []
        for _ in range(count):
            population.append(Genetics._normalize_vector([random.random() - 0.5 for _ in range(weights)]))
        with open('/tmp/generation.txt', 'w') as log:
            for dna in population:
                log.write(json.dumps(dna) + '\n')

    @staticmethod
    def evolve(cycles=10, pop_tests=5, weights=3):
        """Improves AI weights using generic computing"""

        print('Starting time:', time.ctime(), '\nPlanned cycles:', cycles, end='\n\n')
        # read last population
        population = []
        pop_count = 0
        with open('/tmp/generation.txt', 'r') as log:
            for dna in log:
                population.append([json.loads(dna[:-1]), 0])
                pop_count += 1

        # run multiple cycles
        for c in range(cycles):
            seed = random.randrange(1000)
            # test population
            for dna in population:
                fitness = 0
                for j in range(pop_tests):
                    fitness += main(ai=True, vis=False, dna=dna[0], seed=seed + j)
                dna[1] = fitness

            # choose offspring
            offspring = []
            for _ in range(pop_count * 3 // 10):
                pool = sorted(random.sample(population, 4 + pop_count // 20) , key=lambda x: x[1], reverse=True)
                child = [pool[0][0][k] * (pool[0][1] + 1) + pool[1][0][k] * (pool[1][1] + 1) for k in range(weights)]
                child = Genetics._normalize_vector(child)
                # offspring randomly mutates
                if not random.randrange(15):
                    change = random.randrange(weights)
                    child = [(e + (random.random() * 0.4 - 0.2) if i == change else e) for i, e in enumerate(child)]
                    child = Genetics._normalize_vector(child)
                offspring.append([child, 0])

            # replace worst 30% of population with offspring
            population = sorted(population, key=lambda x: x[1], reverse=True)[:pop_count * 7 // 10]
            population += offspring
            # save last population
            with open('/tmp/generation.txt', 'w') as log:
                for dna in population:
                    log.write(json.dumps(dna[0]) + '\n')
            # print status
            print('Finished ', c + 1, '. cycle at ', time.ctime(), '. Best fitness: ', population[0][1], sep='')


class Creator:
    """Seven system random generator"""

    def __init__(self):
        """Sets up initial list of pieces"""
        self.cards = [*LETTERS]
        self.index = 0

    def next(self):
        """Returns next random piece"""
        if self.index == 0: random.shuffle(self.cards)
        out = Tetromino(self.cards[self.index])
        self.index = self.index + 1 if self.index < len(LETTERS) - 1 else 0
        return out


def main(ai=False, vis=True, dna=None, seed=None):
    """Main function; Initiates all components; Contains game loop"""

    assert (ai, vis) != (0, 0)
    dna = dna or [-0.47515, -0.83254, -0.28479]  # default weights
    if seed: random.seed(seed)  # set seed for fair testing
    tetro_count = 0

    screen = Screen(vis)
    if vis: screen.update_message('AI: ON' if ai else 'AI: OFF')
    creator = Creator()
    follow = creator.next()
    keys = (0x71, 0x0a, 0x61, 0x103, 0x105, 0x102, 0x104)

    running = True
    while running:

        # when learning, end game after a few tetrominos
        if not vis and tetro_count == 1000:
            break
        tetro_count += 1

        # remove full rows, decide following tetromino, update score and level
        screen.remove_rows()
        tetro = follow
        follow = creator.next()
        if vis:
            screen.update_stats()
            screen.update_next_tetro(follow)

        # check if game is over
        if not tetro.pos_valid(screen.board):
            running = False
            if vis:
                screen.update_message('GAME OVER!')
                time.sleep(2)

                else:
            if ai:
                # compute best move
                pos, rotation = AI.get_best_move(screen.bit_rep(), dna, tetro, follow)
                for _ in range(rotation):
                    tetro.rotate(screen.board)

                # get tetro into computed position; if path is blocked, move down
                while tetro.pos[0] < pos[0]:
                    if not tetro.step(screen.board, 'RIGHT'): break
                while tetro.pos[0] > pos[0]:
                    if not tetro.step(screen.board, 'LEFT'): break

            if vis:
                # stay in loop while tetromino is movable; handle user input
                step = True
                while step:
                    timer = time.time()
                    while (timer + 0.31 / screen.level) > time.time():
                        key = screen.sdtscr.getch()
                        if key in keys:
                            i = keys.index(key)
                            if i == 0:
                                running = step = False; break  # Q
                            elif i == 1:
                                while tetro.step(screen.board, 'DOWN'): pass  # Enter
                            elif i == 2:
                                ai = not ai; screen.update_message('AI: ON' if ai else 'AI: OFF')  # A
                            elif i == 3:
                                tetro.rotate(screen.board)  # Up
                            elif i == 4:
                                tetro.step(screen.board, 'RIGHT')  # Right
                            elif i == 5:
                                tetro.step(screen.board, 'DOWN')  # Down
                            else:
                                tetro.step(screen.board, 'LEFT')  # Left
                            screen.update_board()
                        time.sleep(0.01)

                    # keep current tetromino under control until no longer movable
                    if not tetro.step(screen.board, 'DOWN'):
                        break
                    screen.update_board()

            else:
                while tetro.step(screen.board, 'DOWN'):
                    pass

    # exit procedure
    if vis:
        screen.cleanup()
        print('Lines cleared:', screen.score)
    else:
        return screen.score


# start game
main()
