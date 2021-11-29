#!/usr/bin/python

"""

Student Names : Gurkirat Kaur
Student IDs (respectively) : 21239534

Note : We are making Gurkirat's Repository as primary repository for submission but it contains commits
from both Anshul and Gurkirat.

Link to Gurkirat's github Repo : https://github.com/gurkirat16/ARC
Link to Anshul's github Repo : https://github.com/multinucliated/ARC

"""

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

'''

###### Task 3 : 0dfd9992 ######

Difficulty : Difficult

After analyzing the task from ARC testing interface, it can be seen that a single row or a single column pattern 
occurs multiple times through out the grid. We read that pattern as a string and store it into a pattern list. 
('insert_pattern_0dfd9992' function below). Patterns are read both rows and column wise ('scan_pattern_0dfd9992' 
function below) because there are some patterns which always contains blanks if read one way (say either row wise 
or column wise).

At some places that single row or column contain blanks (black color blocks). If we find such kind of 
a row/column, we skip over it as it needs to be filled. 

Once all the patterns are identified, we use REGEX to search and fill the patterns for row/columns containing blanks 
using the pattern list. This is carried out in 'get_pattern_0dfd9992' function.

Training Grids solved : All
Testing Grids solved : All

'''

# Create a list to store patterns
pattern = []


# Inserting the patterns identified in rows or columns
def insert_pattern_0dfd9992(a):
    pattern.append(a)


# When looping over missing values, call this to match the required pattern
def get_pattern_0dfd9992(a):
    # Loop over all the existing patterns
    for val in pattern:
        matching = re.search(a, val)  # re.search will find a pattern for missing values (dots)
        if matching:
            return val  # Return the matching pattern
    return a  # In case, pattern is not found in existing list of patterns,
    # it will return the original row and the test/ training case will fail

# Scan the patterns from rows and columns
def scan_pattern_0dfd9992(x_sol):

    row, col = x_sol.shape  # getting dimensions of the grid

    for i in range(row):

        blank = False  # Setting blank flag to decide whether to insert the pattern to list or not
        p = ''  # Intialising pattern for single row

        for j in range(col):
            # If index is blank, set blank to True and skip to new row as current one cannot be added to pattern[]
            if x_sol[i][j] == 0:
                blank = True
                break
            # If blank is yet False, continue creating the pattern by adding the spaces after each color block
            p += str(x_sol[i][j]) + ' '

        # If pattern is created for full row without any blanks, insert the pattern to pattern[]
        if not blank:
            insert_pattern_0dfd9992(p)

# Solving 0dfd9992
def solve_0dfd9992(x):
    # Creating a copy of input grid to work on it
    sol = x.copy()
    # Creating a transpose grid to work on columns
    sol_transpose = np.transpose(sol)
    # getting dimensions of the grid
    row, col = sol.shape

    # Row wise scan
    scan_pattern_0dfd9992(sol)

    # Column wise scan
    scan_pattern_0dfd9992(sol_transpose)

    # Filling out missing part
    for i in range(row):
        blank = False
        p = ''
        for j in range(col):
            # Changing blank flag if there is a blank space
            if sol[i][j] == 0:
                blank = True
                # Filling out blanks with dots (.) because regex can find the matching values from pattern list for
                # the same
                p += '.' + ' '
                continue
            # creating the pattern to get matching values when get_pattern is called
            p += str(sol[i][j]) + ' '

        # If blank we will get the pattern from pattern list,  split it by space and omit the last space
        if blank:
            received_pattern = get_pattern_0dfd9992(p).split(' ')[0:-1]
            # Replacing the row by recieved pattern
            sol[i] = received_pattern

    # Return solution grid
    return sol


'''

###### Task 4 : ded97339 ######

Difficulty : Medium to Difficult

This problem contains individual blocks of blue color in black grids. Task is to join those blocks 
which occur in either same row or same column. We have used Numpy to achieve the result. 

We maintain two copies of the grids (input grid and solution grid). We take all the rows in input grid 
which conatin exactly 2 blue blocks and extract their indices. Then we insert the pattern (blue blocks)
in the solution grid for the same indices. This task is performed in 'insert_pattern_ded97339' function.

The same is done for transposed grid to fill out all the column patterns.

Training Grids solved : All
Testing Grids solved : All

'''

# Inserting blue blocks in solution grid
def insert_pattern_ded97339(inp_grid, sol_grid):
    # Iterating on input and solution rows
    for i_row, s_row in zip(inp_grid, sol_grid):
        # Getting rows where blue color occurs twice
        if np.count_nonzero(i_row) == 2:
            # Getting the starting and ending index
            start, end = np.where(i_row == 8)[0]
            # Updating the black blocks with blue color (Hard coded because only single color is in use)
            # Updating the 'solution grid' always
            s_row[start:end] = 8

# Solving ded97339
def solve_ded97339(x):
    # Creating a copy of input grid to work on it
    sol = x.copy()
    # Creating a transpose of input grid to work on columns
    x_transpose = np.transpose(x)
    # Creating a transpose of solution grid to work on columns
    sol_transpose = np.transpose(sol)

    # Inserting blue blocks row wise
    insert_pattern_ded97339(x, sol)
    # Inserting blue blocks column wise
    insert_pattern_ded97339(x_transpose, sol_transpose)

    # Return solution grid
    return sol


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()
