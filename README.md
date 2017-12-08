# TETRIS

Tetris (Russian: Те́трис) is a tile-matching puzzle video game, originally designed and programmed by
Russian game designer Alexey Pajitnov. It was released on June 6, 1984. He derived its name from the Greek
numerical prefix tetra (all of the game's pieces contain four segments) and tennis, his favorite sport.
(Source: Wikipedia)

You can either play the game yourself or let an AI do the hard work and enjoy watching.
The AI works by trying out all the different possible positions the newly generated tetromino can take in.
It then chooses the best one by evaluating the resulting boards based on weighted computed values (e.g. gaps).
The upcoming tetromino is taken into account, too. Weights were calculated using a generic algorithm.
Tetrominos are chosen by repeatedly shuffling all seven shapes. The resulting list of shapes are the ones
played next. This ensures that no long sequences of the same tetromino appear.
Controls:
            ·-------·
            | turn  |
            | tetro |
            ·-------·
  ·-------· ·-------· ·-------·
  | move  | | move  | | move  |
  | left  | | down  | | right |
  ·-------· ·-------· ·-------·

ENTER:     move tetromino down quickly
Q:         quit game
A:         switch AI on/off

~ idleice 21/05/2017
