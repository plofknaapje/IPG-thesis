# Knapsack Game
The instances associated with the Knapsack Game.

## Instance Format

For each instance, we save a single file _n-m-ins-type.txt_ where
- _n_ is the number of players
- _m_ is the number of variables per player
- _ins_ is either 2, 5, 8, which correspond to a budged of 0.2, 0.5, or 0.8 times the total items weight of each player
- _type_ is either "pot" for the potential instances (i.e., same interaction coefficient for all item of each player), "cij" for positive random interaction coefficients, or "cij-n" for random interaction coefficients.

The format of each file is as follows:
- In the first line we report _n_ and _m_
- In the second line we report the capacitites of each knapsack, in the order of players
- In each following line, we report the propoerties associated with each item. Specifically, the first number is the item number. The next _2n_ entries are the profit and weight of the item for each player (ordered). In the remaing column we report the interaction coefficients of the item for each player with the other players.

### Example of an instance file with 2 players
Assume the instance file contains the following:

```
2 5
47 24
0 17 76 87 12 29 17
1 27 41 16 19 83 41
2 51 2 36 14 67 51
3 23 25 31 21 10 64
4 26 94 27 57 4 80

```

This is equivalent to an instance with _n=2_, _m=5_ where the capacities of the 2 knapsacks are _47_ and _24_, respectively.
The first item (i.e., item _0_), player 1 has a profit of _17_ and a weight of _76_, while player 2 has a profit of _87_ and a weight of _12_. The interaction coefficient of player 1 is _29_, while the one of player 2 is _17_.


### Example of an instance file with 3 players
Consider the following (partial) file:

```
3 5
55 59 62
0 	40 		63 		72 		90 		28 		66 		-1 		46 		-89 	-68 	68 		-90
[...]

```

The meaning of the third row associated with item 0 is:
```
id	p^1_0	w^1_0	p^2_0	w^2_0	p^3_0	w^3_0	C^1_20	C^1_30	C^2_10	C^2_30	C^3_10	C^3_20
```
