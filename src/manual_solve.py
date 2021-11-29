#!/usr/bin/python

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
Difficulty : Hard
ToDo: Add Comments and summary
'''

pattern = []
def insert_pattern_0dfd9992(a):
    pattern.append(a)

def get_pattern_0dfd9992(a):
    for val in pattern :
        matching = re.search(a,val)
        if matching:
            return val
    return a

def solve_0dfd9992(x):

    x_sol = x.copy()
    row, col = x_sol.shape

    # Row wise scanning
    for i in range(row):
        blank = False
        p = ''
        for j in range(col):
            if x_sol[i][j] == 0:
                blank = True
                break
            p += str(x_sol[i][j]) + ' '
            
        # skip to new row
        if blank:
            continue
        else:
            insert_pattern_0dfd9992(p)
            
    # Column wise scanning
    for i in range(row):
        blank = False
        p = ''
        for j in range(col):
            if x_sol[j][i] == 0:
                blank = True
                break
            p += str(x_sol[j][i]) + ' '
            
        if blank:
            continue
        else:
            insert_pattern_0dfd9992(p)

    #Filling out missing part
    for i in range(row):
        blank = False
        p = ''
        for j in range(col):
            if x_sol[i][j] ==  0:
                blank = True
                p += '.' + ' '
                continue
            p += str(x_sol[i][j]) + ' '
        if blank:
            received_pattern = get_pattern_0dfd9992(p).split(' ')[0:-1]
            x_sol[i] = received_pattern

    return x_sol


def insert_pattern_ded97339(inp_matrix, sol_matrix):
    for i_row, s_row in zip(inp_matrix, sol_matrix):
        if np.count_nonzero(i_row) == 2:
            start, end = np.where(i_row == 8)[0]
            s_row[start:end] = 8

def solve_ded97339(x):

    sol = x.copy()
    x_transpose = np.transpose(x)
    sol_transpose = np.transpose(sol)

    insert_pattern_ded97339(x, sol)
    insert_pattern_ded97339(x_transpose, sol_transpose)

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
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
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
        # break
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

