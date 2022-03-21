import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False

# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median_vs_average(nums):
    """
    median takes a non-empty list of numbers,
    returning a boolean of whether the median is
    greater or equal than the average
    If the list has even length, it should return
    the mean of the two elements in the middle.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether median is greater or equal than average.
    
    :Example:
    >>> median_vs_average([6, 5, 4, 3, 2])
    True
    >>> median_vs_average([50, 20, 15, 40])
    False
    >>> median_vs_average([1, 2, 3, 4])
    True
    """ 

    average = (sum(nums)/ len(nums))
    
    length = len(nums)
    sorted_nums = sorted(nums)
    median_index = (length - 1) // 2
    # even length list --> MEDIAN is mean of two middle elements in the middle
    if length % 2 == 0:
        median = (sorted_nums[median_index] + sorted_nums[median_index + 1]) / 2
    # odd length list --> MEDIAN is middle number
    else:
        median = sorted_nums[median_index]
        
    if median >= average:
        #print ("Median is greater than or equal to the average:", median, ">=", average)
        return True
    else:
        #print ("Median is less than the average:", median, "<", average)
        return False
    
# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------
def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance 
    as integers is also i.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    elements as described above.
    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    lst_length = len(ints)
    for i in range(lst_length):
        for j in range(i + 1, lst_length):
            num_difference = abs(ints[i] - ints[j])
            space_difference = j - i
            if space_difference == abs(ints[i] - ints[j]):
                return True
    return False

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Data!', 3)
    'DatDaD'
    >>> n_prefixes('Marina', 4)
    'MariMarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    """
    output = ""
    for i in range(n, 0, -1):
        output += s[0:i]

    return output

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------
def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 4], 2) 
    ['1 2 3 4 5', '2 3 4 5 6']
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    """

    #  zfill() method adds zeros (0) at the beginning of the string, until it reaches the specified length
    
    output = []
    digits = len(str(max(ints)))
    for i in ints:
        temp = str(i).zfill(digits)
        for t in range(1, n + 1):
            temp = str(i - t).zfill(digits) + " " + temp + " " + str(i + t).zfill(digits)
        output.append(temp)
    return output

# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    text = fh.read()
    lines = text.split("\n")
    output = ""
    for line in lines:
        output += line[-1:]
    return output

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """

    index = np.arange(len(A))
    new_index = np.sqrt(index)
    return A + new_index

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """

    return np.equal(np.sqrt(A) % 1, 0)

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """
    diff = np.diff(A)
    growth_rates = np.around(diff/ A[0:-1], 2)

    return growth_rates

def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """
    start = 20
    remainders = start % A
    left_over = np.cumsum(remainders)
    
    try:
        return (left_over > A).tolist().index(True)
    except: 
        return -1

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    
    :param salary: a dataframe of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    # The number of players (num_players)
    try:
        num_players = salary['Player'].nunique()
    except:
        num_players = None
    # The number of teams (num_teams).
    try:
        num_teams = salary['Team'].nunique()
    except:
        num_teams = None
    # The total salary amount over the season (total_salary).
    try:
        total_sal = salary['Salary'].nunique()
    except:
        total_sal = None
    # The name of the player with the highest salary; there are no ties (highest_salary).
    try:
        max_sal = np.max(salary['Salary'])
        highest_sal = salary[salary['Salary'] == max_sal]['Player'].values(0)
    except:
        highest_sal = None
    # The average salary of the Boston Celtics ('BOS'), rounded to the nearest hundredth (avg_bos).
    try:
        avg_bos = np.round(np.mean(salary[salary['Team'] == 'BOS']['Salary']), 2)
    except:
        avg_bos = None
    # The name of player and the name of the team whose salary is the third-lowest,
    # separated by a comma and a space (e.g. John Doe, MIA);
    # if there are ties, return the first based on alphabetical order (third_lowest).
    try:
        third_lwst_nm = salary.sort_values(by = 'Salary')['Player'].values(2)
        third_lwst_tm = salary.sort_values(by = 'Salary')['Team'].values(2)
        third_lwst = third_lowest_nm + ", " + third_lowest_tm
    except:
        third_lwst = None
    lst_nm = lambda x : x.split(' ')[1]
    # Whether there are any duplicate last names (True: yes, False: no), as a boolean
    try:
        if salary['Player'].apply(lst_nm).nunique() == len(salary['Player']):
            duplicate = False
        else:
            duplicate = True
    except:
        duplicate = False
    try:
        hghst_sal_tm = salary[salary['Salary'] == max_sal].values(0)(1)
        total_highest_sal_tm = np.sum(salary[salary['Team'] == hghst_sal_tm]['Salary'])
    except:
        total_highest_sal_tm = None

    indices = ['num_players', 'num_teams', 'total_salary', 'highest_salary', 'avg_bos', 'third_lowest', 'duplicates', 'total_highest']
    data = [num_players, num_teams, total_sal, highest_sal, avg_bos, third_lwst, duplicate, total_highest_sal_tm]
    return pd.Series(data, index = indices)

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).
    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    with open(fp) as file:
        txt = file.read().split('\n')
        entries = [line.split(',') for line in txt]
        for entry in entries:
            if '' in entry:
                entry.remove('')
            if entry == []:
                entries.remove(entry)
        col_dic = {}
        cols = []
        for i in entries[0]:
            col_dic[i] = []
            cols.append(i)
        for entry in entries[1:]:
            for i in range(len(cols)):
                if i < 2:
                    col_dic[cols[i]].append(entry[i].strip('"'))
                elif i < 4:
                    col_dic[cols[i]].append(float(entry[i].strip('"')))
                else:
                    col_dic[cols[i]].append(entry[i].strip('"') + ',' + entry[i + 1].strip('"'))
    return pd.DataFrame(col_dic)

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median_vs_average'],
    'q02': ['same_diff_ints'],
    'q03': ['n_prefixes'],
    'q04': ['exploded_numbers'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['salary_stats'],
    'q08': ['parse_malformed']
}

def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True