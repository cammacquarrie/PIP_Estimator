# tSIERA_Estimator
Neural Network used to estimate Pitches per Inning Pitched
based off coursera project 
https://github.com/MaxPoon/coursera-Advanced-Machine-Learning-specialization/blob/master/Introduction-To-Deep-Learning/week2/my1stNN.ipynb

Training:
https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=100&type=c,48,105,106,108,109,111,113,112,120,121,31,13&season=2018&month=0&season1=2002&ind=1&team=0&rost=0&age=0&filter=&players=0

2018 Season:
https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg=all&qual=100&type=c,48,105,106,108,109,111,113,112,120,121,31,13&season=2018&month=0&season1=2002&ind=1&team=0&rost=0&age=0&filter=&players=0

Goal of the project was to use statitics that quickly normalize to estimate SIERA accurately over as short of a period as possible.

SIERA is currently the best available true ERA reflector so it was chosen as the target.

Inputs were:

-Pitch selection by % (FB%, SL%, CT%, CB%, CH%, SF%, KN%, XX%)

-Pitch velocities (FBv, SLv, CTv, CBv, CHv, SFv, KNv)

-Contact%

-O-Swing%

-Z-Swing%

-Zone%


NN was organized as follows:

Layer 1 (Input): 19x100

Layer 2 (Dense, relu): 100x100

Layer 3 (Dense, relu): 100x50

Layer 4 (Dense, relu, Output): 50x1


RMSE for Test between 0.07-0.12


DATA - All data was taken from Fangraphs.com:

Pitcher season values 2002-2018 with minimum 30IP

- 5477 data points (4500 training set, 977 test set. 82/18 split)

- Learning rate set at 0.0001


TODO:

- Try SwStr instead of Contact

- Try to split Contact into O-contact, Z-Contact

- Add option to export NN for permanance
