# GoDeep


concerns atm:
1. Saving
2. Constant error (pack error, with convert to tensor, it's some dtype problem)

TODO:
1. Need to refine the self_play process, make it return a list of positions and use the model to generate them
2. Write helper functions (probably inside model) so that it takes positions, not the weird tuples we have been giving it
3. These changes could solve our dtype problems (maybe)
4. Write train policy, should be easy once the self-play stuff has been fixed. 