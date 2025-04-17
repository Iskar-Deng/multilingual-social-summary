#!/bin/bash

TEXT="The Seattle Mariners pulled off a dramatic comeback last night against the Houston Astros. Trailing 4-1 going into the eighth inning, the Mariners rallied with a pair of home runs from Julio Rodríguez and Cal Raleigh to tie the game. In the bottom of the ninth, Ty France delivered a walk-off single to give Seattle a thrilling 5-4 victory. The bullpen was stellar, shutting down the Astros' offense over the final three innings. With the win, the Mariners improve their record to 10-5 and remain in second place in the AL West standings."

# Run the model inference
python src/model_train/model_test.py "$TEXT"
