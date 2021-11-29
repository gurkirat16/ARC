#!/usr/bin/python

"""

Student Names : Gurkirat Kaur, Anshul Verma
Student IDs (respectively) : 21239534, 21235667

Note : We are making Gurkirat's Repository as primary repository for submission but it contains commits
from both Anshul and Gurkirat.

Link to Gurkirat's github Repo : https://github.com/gurkirat16/ARC
Link to Anshul's github Repo : https://github.com/multinucliated/ARC

"""

import json
import os
import re

import numpy as np

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

"""
                                    TASK 1 :: c8cbb738 :: Hard Level Difficulty 

When we've looked in to the picture, it seems like :
1) We need to create the output grid of max distance which is present in the pattern Eg: Square, Plus sign , 2 types 
of rectangle
2) Once final output grid is initialized, we need to extract the each unique color pattern form the main data.
3) Once those are extracted, we'll make the background color to zero (why zero? we'll see this 4th point)   
4) When we'll hold the extracted data, we will sum all the pattern that we have extracted , its a simple matrix addition
    as we have made the background zero, so it can be added without any extra numbers. 
Additionally, there are few more condition , where I've added the padding for the rectangles accordingly     
                                **********************************************
                                *        Training Grids solved : All         *
                                *         Testing Grids solved : All         *
                                **********************************************
"""


def solve_c8cbb738(data):
    """
    This function will dp all the required calculation which we have just explained above

    @param data : input numpy array grid values
    return : output grid with the correct values
    """
    # getting the unique color with their total count in the input data
    num, count = np.unique(data, return_counts=True)

    # converting those values to list
    num = list(num)
    count = list(count)

    # getting the max count from the count list
    max_ = max(count)

    # this loop will look for the max value index and break , i will hold the max value index position
    i = 0
    for _temp in count:
        if max_ == _temp:
            break
        i += 1

    # removing that index (basically removing the background color index and counts) and saving the background color
    # to a new variable
    count.pop(i)
    background = num[i]
    num.pop(i)

    low = []
    high = []

    # getting the min and max value from the x co-coordinates value for each color pattern, this will help us to decide
    # the output grid size
    for val in num:
        x, y = np.where(data == val)
        low.append(min(x))
        high.append(max(x))

    # once we have those min and max list values, we can subtract and get the highest value, so that best value will
    # contributing to output grid shape
    best = 0
    for x, y in list(zip(low, high)):
        _temp = y - x
        if _temp > best:
            best = _temp

    # as we have seen in the ARC, the logic says, we need to add 1 for the best grid
    best += 1

    # now most important : we are replacing the background color with 0 so that its easy to search and add the
    # pattern later, when' will be able to find it
    data = np.where(data == background, 0, data)

    # This loop is most important one which consist of serveral task mentioned below:
    # 1) As it loop , each color values from the main data
    # 2) it tries to get the  x and y values for those color pattern from the "data" by doing the slicing concept
    # 3) it explicitly replaces the extra color number if present in the extracted data that is temp_data_
    # 4) we are checking for one condition, if the extracted shape is not of final grid shape which means it is a
    # rectangle, so we are adding the pattern accordingly
    #     if row is less than the best : we are adding the padding to left and right
    #     if column is less then the best :  we are adding the padding to upper and lower
    # 5) Appending those values to a list called as shape_area

    shape_area = []
    for val in num:
        # for every unique value we are getting the x y values
        x, y = np.where(data == val)

        # extracting those values from the main data by using the slicing concept
        temp_data_ = data[min(x): max(x) + 1, min(y): max(y) + 1]

        # removing the extra color, if present which will replace it be the 0
        temp_data_ = np.where(temp_data_ != val, 0, temp_data_)

        # checking for the condition, if that shape is not equal to the best shape then that means that the extracted
        # shape is rectangle, so for every rectangle or uneven shape this condition will get true
        if temp_data_.shape != tuple((best, best)):

            # checking for a type of rectangle, if row is less then this condition will be true
            if temp_data_.shape[0] <= best and temp_data_.shape[1] == best:
                # this will help us to get how much padding we have to do, which will be stored in q
                # this will help us to add the data on the upper and lower side of adding to  a rectangle
                q = np.zeros([(temp_data_.shape[1] - temp_data_.shape[0]) // 2, best])
                # upper padding added
                append_val = np.append(q, temp_data_)
                # lower padding added
                append_val = np.append(append_val, q)
                # reshaping back to the required grid size
                append_val = append_val.reshape(best, best)

            # checking for a type of rectangle, if column is less then this condition will be true
            if temp_data_.shape[0] == best and temp_data_.shape[1] <= best:
                # this will help us to get how much padding we have to do, which will be stored in q
                # this will help us to add the data on the left and right  side of adding to  a rectangle
                q = np.zeros([best, (temp_data_.shape[0] - temp_data_.shape[1]) // 2])
                # left padding added
                append_val = np.append(q, temp_data_, axis=1)
                # right padding added
                append_val = np.append(append_val, q, axis=1)
                # reshaping back to the required grid size
                append_val = append_val.reshape(best, best)

            shape_area.append(append_val)
        else:
            # if condition is false, we are directly adding those values
            shape_area.append(temp_data_)

    # adding extracted values to a grid of [best x best] which consist of  zeros
    _temp_array = np.zeros([best, best])
    for every_shape in shape_area:
        _temp_array = np.add(_temp_array, every_shape)

    # and once its all added up, we are finally replacing back to the background value
    x = np.where(_temp_array == 0, background, _temp_array)

    # returning the output grid
    return x


"""
                                    TASK 2 :: 9f236235 :: Hard Level Difficulty 

When we've looked in to the picture, it seems like :
1) We need to create the output grid of size , where we need to find the color block size and loop for that size in the 
    whole dataset which will help us to create the final grid output  
2) We need to find the best block size for color pattern which is inscribed in the horizontal and vertical lines 
3) Lets suppose, we got the best size of the color pattern, we will loop over each datapoint with the window size of 
   the best values.
4) For every iteration, we will capture the color pallet 
5) From collected color pallet, will draw a 2nd last matrix 
6) And finally, we will flip the the matrix which will be our output grid 
 
                                **********************************************
                                *        Training Grids solved : All         *
                                *         Testing Grids solved : All         *
                                **********************************************
"""


def solve_9f236235(data):
    """
        This function will do all the required calculation which we have just explained above

        @param data : input numpy array grid values
        return : output grid with the correct values
    """
    # getting the initial color values
    val_0_0 = data[0][0]

    best_grid = None
    # we will iterate over every data from 1 to row size + 1 ; we are doing this because we are multiply the values in
    # the given logic below
    for i in range(1, data.shape[0] + 1):
        # creating the 2D array for the value of val_0_0 for every number of i
        temp_grid = np.array([[val_0_0] * i] * i)

        # getting the actual data for the given i from the data
        data_grid = data[:i, :i]

        # checking, if the 2D array is equal or not, if true, it will assign the best value which is nothing but the
        # shape value
        if np.array_equal(temp_grid, data_grid):
            best_grid = tuple((i, i))

    val = []
    # as we know the best grid value, so we will iterate over data for the given value of best ie will will
    # search for best x best in the loop

    # the first for loop will [best x best] will look in the x axis with counter step of 1 as we have the horizontal
    # lines
    for x in range(1, data.shape[0] + 1, best_grid[0] + 1):
        _temp = []

        # the second for loop will [best x best] will look in the y axis with counter step of 1 as we have
        # the vertical lines

        for y in range(1, data.shape[0] + 1, best_grid[0] + 1):
            # extracting the color pattern for the best values
            _temp_val = data[x:x + best_grid[0] - 1, y: y + best_grid[1] - 1]
            # extracting the color from the pattern and storing it in the list
            _temp.append(_temp_val[0][0])

        # storing all the values of y in the val list
        val.append(_temp)

    # converting it to the numpy array
    val = np.array(val)

    # flipping that array at the column level
    x = np.flip(val, 1)

    # returning the final flipped value as a output
    return x


'''

                                    Task 3 :: 0dfd9992 :: Hard Difficulty

After analyzing the task from ARC testing interface, it can be seen that a single row or a single column pattern 
occurs multiple times through out the grid. We read that pattern as a string and store it into a pattern list. 
('insert_pattern_0dfd9992' function below). Patterns are read both rows and column wise ('scan_pattern_0dfd9992' 
function below) because there are some patterns which always contains blanks if read one way (say either row wise 
or column wise).

At some places that single row or column contain blanks (black color blocks). If we find such kind of 
a row/column, we skip over it as it needs to be filled. 

Once all the patterns are identified, we use REGEX to search and fill the patterns for row/columns containing blanks 
using the pattern list. This is carried out in 'get_pattern_0dfd9992' function.

                                **********************************************
                                *        Training Grids solved : All         *
                                *         Testing Grids solved : All         *
                                **********************************************

'''

# Create a list to store patterns
pattern = []


def insert_pattern_0dfd9992(a):
    """
        Inserts the patterns identified in rows or columns
        I/P : Pattern String
        O/P : None (Pattern appended in list)
    """
    pattern.append(a)


def get_pattern_0dfd9992(a):
    """
        When looping over missing values, call this to match the required pattern
        I/P : Pattern String
        O/P : Matched pattern with filled values
    """
    # Loop over all the existing patterns
    for val in pattern:
        matching = re.search(a, val)  # re.search will find a pattern for missing values (dots)
        if matching:
            return val  # Return the matching pattern
    return a  # In case, pattern is not found in existing list of patterns,
    # it will return the original row and the test/ training case will fail


def scan_pattern_0dfd9992(x_sol):
    """
        Scans the patterns from rows and columns. Inserts into pattern list if pattern is identified.
        I/P : Solution grid
        O/P : None
    """

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


def solve_0dfd9992(x):
    """
       Main function to solve the task. Explanation same as in description
       I/P : Problem grid (2D Numpy array)
       O/P : Solved grid with required values
    """
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

                                Task 4 :: ded97339 :: Medium to Hard Difficulty


This problem contains individual blocks of blue color in black grids. Task is to join those blocks 
which occur in either same row or same column. We have used Numpy to achieve the result. 

We maintain two copies of the grids (input grid and solution grid). We take all the rows in input grid 
which conatin exactly 2 blue blocks and extract their indices. Then we insert the pattern (blue blocks)
in the solution grid for the same indices. This task is performed in 'insert_pattern_ded97339' function.

The same is done for transposed grid to fill out all the column patterns.

                                **********************************************
                                *        Training Grids solved : All         *
                                *         Testing Grids solved : All         *
                                **********************************************

'''


def insert_pattern_ded97339(inp_grid, sol_grid):
    """
        Updates the solution grid. Inserts blue blocks in the required positions in solution grid.
        I/P : problem grid and solution grid (initially just the copy of problem grid)
        O/P : None
    """
    # Iterating on input and solution rows
    for i_row, s_row in zip(inp_grid, sol_grid):
        # Getting rows where blue color occurs twice
        if np.count_nonzero(i_row) == 2:
            # Getting the starting and ending index
            start, end = np.where(i_row == 8)[0]
            # Updating the black blocks with blue color (Hard coded because only single color is in use)
            # Updating the 'solution grid' always
            s_row[start:end] = 8


def solve_ded97339(x):
    """
       Main function to solve the task. Explanation same as in description
       I/P : Problem grid (2D Numpy array)
       O/P : Solved grid with required values
    """

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

"""

Summary/Reflection :

We thought of doing this code using the Machine Learning/ Deep learning and genetic algorithm and the Author 
of that paper also suggests the same. ML/DL approaches try to understand the rules for the different patterns 
at once which can help solve all problems of ARC. Some Kaggle masters have got the accuracy of 20%-30% using 
these approaches. But in our case the task was to solve 3 individual problems and solving them using ML/DL 
approaches as compared to Rule-Based approach would be inefficient. 

For all the individual grids solved manually, we have used Python libraries, such as Numpy (predominantly) and Regex. 
These libraries try to learn rules/patterns of individual grid and then give the solution of testing grids based 
on the learnings. Numpy contains important functions such as np.where, np.transpose, np.unique, np.count_nonzero which help
to make the task easier. Regex also helped in pattern searching and matching (re.search). Rest of the repository uses 
loops, branches and python datastructures such as lists, tuples, np arrays etc.

According to us, Rule based approach (bottom to top) and ML/DL approaches (top to bottom) can be combined to generate
an effective solution for ARC problems.


"""
