# The Abstraction and Reasoning Corpus (ARC)

This repository contains the ARC task data, as well as a browser-based interface for humans to try their hand at solving
the tasks manually.

*"ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric
intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like
form of general fluid intelligence."*

A complete description of the dataset, its goals, and its underlying logic, can be found
in: [The Measure of Intelligence](https://arxiv.org/abs/1911.01547).

As a reminder, a test-taker is said to solve a task when, upon seeing the task for the first time, they are able to
produce the correct output grid for *all* test inputs in the task (this includes picking the dimensions of the output
grid). For each test input, the test-taker is allowed 3 trials (this holds for all test-takers, either humans or AI).

## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (400 tasks). Use these to prototype your algorithm or to train
  your algorithm to acquire ARC-relevant cognitive priors.
- `data/evaluation`: contains the task files for evaluation (400 tasks). Use these to evaluate your final algorithm. To
  ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking
  at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its
  evaluation score as feedback).

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid
size is 1x1 and the largest is 30x30.

When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input(s) of the
test pair(s). The goal is to construct the output grid(s) corresponding to the test input grid(s), using 3 trials for
each test input. "Constructing the output grid" involves picking the height and width of the output grid, then filling
each cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only *exact* solutions (
all cells match the expected answer) can be said to be correct.

## Usage of the testing interface

The testing interface is located at `apps/testing_interface.html`. Open it in a web browser (Chrome recommended). It
will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the
current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top
  left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some
  modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the
      selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After
      copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the
      cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells"
  are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 3-trials
rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for
the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.

## Purpose of the Repository

Purpose of this repository is to manually solve the ARC tasks using Python. We have used multiple approaches/libraries
to solve 4 problems like Numpy, Regex etc. Problems are of difficulty level - Difficult and Medium to difficult (choosen
from "`data/training`" directory). The solutions are generic for same grid patterns.

### Task 1 `solve_c8cbb738` :

Here, the program tries to find and store the patterns by scanning rows and columns. It tried to find the different
pattern with different colors. Check the max distance between the single pattern for the best output grid and finally
creates those patterns

![img_4.png](img_4.png)

### Task 2 `solve_9f236235` :

The program tries to find and store the patterns by scanning rows and columns. It tries to find the best color pallet
size and then iterate over the main data by taking the best shape as the window size. For every window size it take the
color pallet and stores them and Finally we will flip that matrix and get the required output

![img_3.png](img_3.png)

### Task 3 `solve_0dfd9992` :

Here, the program tries to find and store the patterns by scanning rows and columns. For all rows with missing blocks,
it tries to match a pattern from accumulated patterns using Regex. Once it gets a match, it updates the row with the
matching pattern.

![img.png](img.png)

### Task 4 `solve_ded97339` :

Here, the program gets all the rows containing 2 blue color blocks using Numpy. Then it gets the start and end indices
of blue blocks and replaces all the black blocks within the indices by blue blocks.

![img_1.png](img_1.png)

## Libraries used

- Numpy
- Regex

## Summary (Including few extra thoughts)

We thought of doing this code using the Machine Learning/ Deep learning and genetic algorithm and the Author of that
paper also suggests the same. ML/DL approaches try to understand the rules for the different patterns at once which can
help solve all problems of ARC. Some Kaggle masters have got the accuracy of 20%-30% using these approaches. But in our
case the task was to solve 3 individual problems and solving them using ML/DL approaches as compared to Rule-Based
approach would be inefficient.

For all the individual grids solved manually, we have used Python libraries, such as Numpy (predominantly) and Regex.
These libraries try to learn rules/patterns of individual grid and then give the solution of testing grids based on the
learnings. Numpy has helped us a lot in the vertorization, searching , replacing etc where it contains important
functions such as np.where, np.transpose, np.unique, np.flip , np.count_nonzero which help to make the task easier.
Regex also helped in pattern searching and matching (re.search). Rest of the repository uses loops, branches and python
datastructures such as lists, tuples, np arrays etc.

According to us, Rule based approach (bottom to top) and ML/DL approaches (top to bottom) can be combined to generate an
effective solution for ARC problems.

In our perspective, we think this is a good start to think how human actually think and approach to the problems like
mentioned in the ARC. When we were solving those problems and saw any image for the first time, we as a human start
augmenting the data accordingly and search for possible combination and thats' hwo we have solved. but to solve these
using machine we need some combination of ML/DL and genetic algorithms.