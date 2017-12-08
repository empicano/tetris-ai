# TETRIS

Tetris (Russian: Те́трис) is a tile-matching puzzle video game, originally designed and programmed by Russian game designer Alexey Pajitnov. It was released on June 6, 1984. He derived its name from the Greek numerical prefix tetra and tennis, his favorite sport. 

Source: Wikipedia

## The Game

You can play the game normally, but the exciting thing about this version is the implemented AI, which can play the game on its own forever. The AI works by choosing the best of all the possible positions the new tetromino can take in by evaluating the resulting boards based on weighted values (e.g. number of gaps). The following tetromino is taken into account, too. Weights were calculated using a generic algorithm.

Tetrominos are chosen by shuffling the seven shapes. The resulting list of shapes are the ones played next. This ensures that no long sequences of the same tetromino appear.

## Controls:

Start the game by running *python3 tetris.py* in your terminal.

Key | Function
----|---------
⇧ | turn clockwise
⇦ | move to the left
⇨ | move to the right
⇩ | move one block down
↵ | move all the way down
Q | quit game
A | switch AI on/off
