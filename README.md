# TETRIS

Tetris (Russian: Те́трис) is a tile-matching puzzle video game, originally designed and programmed by Russian game designer Alexey Pajitnov. It was released on June 6, 1984. He derived its name from the Greek numerical prefix tetra (all of the game's pieces contain four segments) and tennis, his favorite sport. 

Source: Wikipedia

***

The exciting thing about this version is that i implemented an AI, which can play the game on its own forever.

The AI works by choosing the best one of all the possible positions the new tetromino can take in by evaluating the resulting boards based on weighted values (e.g. number of gaps). The following tetromino is taken into account, too. Weights were calculated using a generic algorithm.

Tetrominos are chosen by shuffling the seven shapes. The resulting list of shapes are the ones played next. This ensures that no long sequences of the same tetromino appear.

***

Controls:
<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·-------·<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| turn  |<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| tetro |<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;·-------·<br />
&nbsp;&nbsp;  ·-------· ·-------· ·-------·<br />
&nbsp;&nbsp;  | move  | | move  | | move  |<br />
&nbsp;&nbsp;  | left  | | down  | | right |<br />
&nbsp;&nbsp;  ·-------· ·-------· ·-------·<br />
<\p>

ENTER:     move tetromino down quickly
Q:         quit game
A:         switch AI on/off

~ idleice 21/05/2017
