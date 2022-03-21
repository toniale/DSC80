import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def data_load(scores_fp):
    """
    follows different steps to create a dataframe
    :param scores_fp: file name as a string
    :return: a dataframe
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> list(scores.columns)
    ['attempts', 'highest_score']
    >>> isinstance(scores.index[0], int)
    False
    """
    # a
    df = pd.read_csv(scores_fp, usecols = ['name', 'tries', 'highest_score', 'sex'])

    # b
    df = df.drop(columns = ['sex'])

    # c
    df = df.rename(columns = {'name':'firstname', 'tries':'attempts'})

    # d
    df = df.set_index('firstname')

    return df


def pass_fail(scores):
    """
    modifies the scores dataframe by adding one more column satisfying
    conditions from the write up.
    
    :param scores: dataframe from the question above
    :return: dataframe with additional column pass
    
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> len(scores.columns)
    3
    >>> scores.loc["Julia", "pass"]=='Yes'
    True

    """
    
#     "Yes" if a number of attempts is strictly less than 3 and the score is >= 50
#     "Yes" if a number of attempts is strictly less than 6 and the score is >= 70
#     "Yes" if a number of attempts is strictly less than 10 and the score is >= 90
#     "No" otherwise
    
    def p_np(attempts, score):
        if attempts < 3 and score >= 50:
            return "Yes"
        elif attempts < 6 and score >= 70:
            return "Yes"
        elif attempts < 10 and score >= 90:
            return "Yes"
        else: 
            return "No"
        
    scores['pass'] = list(map(p_np, scores['attempts'], scores['highest_score']))
    
    return scores


def av_score(scores):
    """
    returns the average score for those students who passed the test.
    
    :param scores: dataframe from the second question
    :return: average score
    
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> av = av_score(scores)
    >>> isinstance(av, float)
    True
    >>> 91 < av < 92
    True
    """

    return np.mean(scores[scores['pass'] == 'Yes']['highest_score'])


def highest_score_name(scores):
    """
    finds the highest score and people who received it
    
    :param scores: dataframe from the second question
    :return: dictionary where the key is the highest score and the value(s) is a list of name(s)
    
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> highest = highest_score_name(scores)
    >>> isinstance(highest, dict)
    True
    >>> len(next(iter(highest.items()))[1])
    3
    """
    dictionary = {}
    keys = np.max(scores['highest_score'])
    values = list(scores[scores['highest_score'] == keys].index)
    dictionary[keys] = values
    return dictionary

def idx_dup():
    """
    Answers the question in the write up.
    Is it possible for a dataframe's index to have duplicate values?
    
    :return:
    >>> ans = idx_dup()
    >>> isinstance(ans, int)
    True
    >>> 1 <= ans <= 6
    True
    """
    # Yes and index values are not restricted to integers
    return 6


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def trick_me():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  trick_me()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    tricky_1 = pd.DataFrame({'Name': ['Tonia', 'Jus', 'Mar', 'Jan', 'Steve'],
             'Name': ['Le', 'Eld', 'Lang', 'Tief', 'Lev'],
             'Age': ['20', '30', '37', '35', '36']})
    tricky_1.to_csv('tricky_1.csv', index = False)
    trick_2 = pd.read_csv('tricky_1.csv')
    return 'C'



def reason_dup():
    """
     Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_dup()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    return 'B'



def trick_bool():
    """
     Answers the question in the write-up
    :return: a list with three letters
    >>> ans =  trick_bool()
    >>> isinstance(ans, list)
    True
    >>> isinstance(ans[1], str)
    True

    """
    return ['D', 'B', 'M']

def reason_bool():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_bool()
    >>> ans == 'A' or ans == 'B' or ans == "C" or ans =="D"
    True

    """
    return 'B'


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def change(x):
    """
    Returns 'MISSING' when x is `NaN`,
    Otherwise returns x
    >>> change(1.0) == 1.0
    True
    >>> change(np.NaN) == 'MISSING'
    True
    """
    #if x == np.NaN:
    if np.isnan(x):
        return "MISSING"
    else:
        return x

def correct_replacement(nans):
    """
    changes all np.NaNs to "Missing"
    
    :param nans: given dataframe
    :return: modified dataframe
    >>> nans = pd.DataFrame([[0,1,np.NaN], [np.NaN, np.NaN, np.NaN], [1, 2, 3]])
    >>> A = correct_replacement(nans)
    >>> (A.values == 'MISSING').sum() == 4
    True
    """
    return nans.applymap(change)


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def population_stats(df):
    """
    population_stats which takes in a dataframe df
    and returns a dataframe indexed by the columns
    of df, with the following columns:
    
        - `num_nonnull` contains the number of non-null
          entries in each column,
          
        - `pct_nonnull` contains the proportion of entries
          in each column that are non-null,
          
        - `num_distinct` contains the number of distinct
          entries in each column,
          
        - `pct_distinct` contains the proportion of (non-null)
          entries in each column that are distinct from each other.

    :Example:
    >>> data = np.random.choice(range(10), size=(100, 4))
    >>> df = pd.DataFrame(data, columns='A B C D'.split())
    >>> out = population_stats(df)
    >>> out.index.tolist() == ['A', 'B', 'C', 'D']
    True
    >>> cols = ['num_nonnull', 'pct_nonnull', 'num_distinct', 'pct_distinct']
    >>> out.columns.tolist() == cols
    True
    >>> (out['num_distinct'] <= 10).all()
    True
    >>> (out['pct_nonnull'] == 1.0).all()
    True
    """
    col_names = ['index',"num_nonnull", 'pct_nonnull','num_distinct','pct_distinct']
    cols = df.columns
    num_nonnull = df.count(axis = 0)
    pct_nonnull = num_nonnull / len(df.index)
    num_distinct = df.nunique(axis = 0, dropna = True)
    pct_distinct = num_distinct / num_nonnull
    data =[cols, num_nonnull,pct_nonnull,num_distinct,pct_distinct]
    out_df = pd.DataFrame()
    ## loop through col
    for i in range(len(col_names)):
        out_df[col_names[i]] = np.array(data[i])   
    out_df = out_df.set_index('index')
    return out_df


def most_common(df, N=10):
    """
    `most_common` which takes in a dataframe df and returns
    a dataframe of the N most-common values (and their counts)
    for each column of df.

    :param df: input dataframe.
    :param N: number of most common elements to return (default 10)
.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 2))
    >>> df = pd.DataFrame(data, columns='A B'.split())
    >>> out = most_common(df, N=3)
    >>> out.index.tolist() == [0, 1, 2]
    True
    >>> out.columns.tolist() == ['A_values', 'A_counts', 'B_values', 'B_counts']
    True
    >>> out['A_values'].isin(range(10)).all()
    True
    """
    out_df = pd.DataFrame()
    for col in df.columns:
        temporary_df = df.groupby(col).size().sort_values(ascending = False)
        if len(temporary_df.index) < N:
            temp = np.array([np.NaN] * (N - len(temporary_df.index)))
            values = np.append(temporary_df.index, temp)
            counts = np.append(temporary_df.values, temp)
        else:
            values = np.array(temporary_df.index)[:N]
            counts = np.array(temporary_df.values)[:N]
        out_df[col + '_values'] = values
        out_df[col + '_counts'] = counts
    return out_df


# ---------------------------------------------------------------------
# Question 5
# ---------------------------------------------------------------------


def null_hypoth():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hypoth()).issubset({1,2,3,4})
    True
    """
    return [1]


def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0,1]).all()
    True
    """
    y = np.random.binomial(300, .99)
    return np.array(y * [1] + (300 - y) * [0])


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """
    p_vals = []
    for i in range(N):
        sample = simulate_null()
        p_vals.append(np.count_nonzero(sample))
    non_faulty = 292 # 8 complaints
    est_p_val = np.count_nonzero(np.array(p_vals) < non_faulty) / N
    return est_p_val


# ---------------------------------------------------------------------
# Question 6
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    """
    `super_hero_powers` takes in a dataframe like
    powers and returns a list with the following three entries:
        - The name of the super-hero with the greatest number of powers.
        - The name of the most common super-power among super-heroes whose names begin with 'M'.
        - The most pobpular super-power among those with only one super-power.

    :Example:
    >>> fp = os.path.join('data', 'superheroes_powers.csv')
    >>> powers = pd.read_csv(fp)
    >>> out = super_hero_powers(powers)
    >>> isinstance(out, list)
    True
    >>> len(out)
    3
    >>> all([isinstance(x, str) for x in out])
    True
    """
    # The name of the super-hero with the greatest number of powers.
    num_powers = (powers.drop('hero_names', axis=1).applymap(np.count_nonzero)).sum(axis=1).values
    most_powers = max(num_powers)
    # add column containing number of powers
    powers['total_powers_num'] = num_powers
    greatest_powers_superhero = powers[powers['total_powers_num'] == most_powers]['hero_names'].values[0]

    
    # The name of the most common super-power among super-heroes whose names begin with 'M'.
    m_names = (powers[powers['hero_names'].str.startswith('M')]).drop('hero_names', axis=1).applymap(np.count_nonzero)
    sum_m_pows = (m_names.sum(axis = 0))
    sorted_sum_m_pows = sorted(sum_m_pows)[-2]
    most_com_m = sum_m_pows[sum_m_pows == sorted_sum_m_pows].idxmax()

    # The most pobpular super-power among those with only one super-power.
    one_pow = powers[powers['total_powers_num']==1]
    one_pow_sum = one_pow.drop('hero_names', axis=1).applymap(np.count_nonzero).sum(axis=0)
    sorted_sum = sorted(one_pow_sum)[-2]
    common_one_power = one_pow_sum[one_pow_sum == sorted_sum].idxmax()
    common_one_power
    
    return [greatest_powers_superhero, most_com_m, common_one_power]

# ---------------------------------------------------------------------
# Question 7
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    """
    clean_heroes takes in the dataframe heroes
    and replaces values that are 'null-value'
    place-holders with np.NaN.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = clean_heroes(heroes)
    >>> out['Skin color'].isnull().any()
    True
    >>> out['Weight'].isnull().any()
    True
    """
    def replace_place_holders(x):
        if x == '-':
            return np.NaN
        elif x == -99:
            return np.NaN
        else:
            return x
        
    return heroes.applymap(replace_place_holders)

  

def super_hero_stats():
    """
    Returns a list that answers the questions in the notebook.
    :Example:
    >>> out = super_hero_stats()
    >>> out[0] in ['Marvel Comics', 'DC Comics']
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    >>> out[3] in ['good', 'bad']
    True
    >>> isinstance(out[4], str)
    True
    >>> 0 <= out[5] <= 1
    True
    """
#     Which publisher has a greater proportion of 'bad' characters -- Marvel Comics or DC Comics?
#     Give the number of characters that are NOT human, or the publisher is not Marvel Comics nor DC comics. 
#     For this question, only consider race "Human" as human, races such as "Human / Radiation" don't count as human.
#     Give the name of the character that's both greater than one standard deviation above mean in height and at least one standard deviation below the mean in weight.
#     Who is heavier on average: good or bad characters?
#     What is the name of the tallest Mutant with no hair?
#     What is the probability that a randomly chosen 'Marvel' character in the dataset is a woman?

    return ['Marvel Comics', 531, "Groot", "bad", "Onslaught", 111/388 ]

# ---------------------------------------------------------------------
# Question 8
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    """
    `bhbe` ('blond-hair-blue-eyes') returns a boolean
    column that labels super-heroes/villains that
    are blond-haired *and* blue eyed.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = bhbe_col(heroes)
    >>> isinstance(out, pd.Series)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out.sum()
    93
    """
    cleaned_heroes = clean_heroes(heroes)
    cleaned_heroes["Hair color"] = cleaned_heroes['Hair color'].str.lower()
    blond = (cleaned_heroes["Hair color"].values == "blond") | (cleaned_heroes["Hair color"].values == "strawberry blond")
    
    blue_eyes = cleaned_heroes["Eye color"].str.contains("blue")
    return pd.Series(blond & blue_eyes)

def observed_stat(heroes):
    """
    observed_stat returns the observed test statistic
    for the hypothesis test.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = observed_stat(heroes)
    >>> 0.5 <= out <= 1.0
    True
    """
    heroes['blonde_blue'] = bhbe_col(heroes)
    blond_and_blue = heroes[heroes['blonde_blue'] == True]
    return blond_and_blue[blond_and_blue['Alignment']=='good'].shape[0] / blond_and_blue.shape[0]

def simulate_bhbe_null(n):
    """
    `simulate_bhbe_null` that takes in a number `n`
    that returns a `n` instances of the test statistic
    generated under the null hypothesis.
    You should hard code your simulation parameter
    into the function; the function should *not* read in any data.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = simulate_bhbe_null(10)
    >>> isinstance(out, pd.Series)
    True
    >>> out.shape[0]
    10
    >>> ((0.45 <= out) & (out <= 1)).all()
    True
    """
#   probability = observed_stat(heroes)
    probability = 0.85
    have_both_num = 93
    simulations = []
    for i in range(n):
        test_stat = np.random.binomial(have_both_num, probability) / have_both_num
        simulations.append(test_stat)
    return pd.Series(simulations)


def calc_pval():
    """
    calc_pval returns a list where:
        - the first element is the p-value for
        hypothesis test (using 100,000 simulations).
        - the second element is Reject if you reject
        the null hypothesis and Fail to reject if you
        fail to reject the null hypothesis.

    :Example:
    >>> out = calc_pval()
    >>> len(out)
    2
    >>> 0 <= out[0] <= 1
    True
    >>> out[1] in ['Reject', 'Fail to reject']
    True
    """
    obs_stat_heroes = 0.85
    p_val = np.count_nonzero(simulate_bhbe_null(100_000) > obs_stat_heroes) / 100_000
    if p_val > .01:
        return [p_val, 'Reject']
    else:
        return [p_val, 'Fail to Reject']
   


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['data_load', 'pass_fail', 'av_score',
            'highest_score_name', 'idx_dup'],
    'q02': ['trick_me', 'reason_dup', 'trick_bool', 'reason_bool'],
    'q03': ['change', 'correct_replacement'],
    'q04': ['population_stats', 'most_common'],
    'q05': ['null_hypoth', 'simulate_null', 'estimate_p_val'],
    'q06': ['super_hero_powers'],
    'q07': ['clean_heroes', 'super_hero_stats'],
    'q08': ['bhbe_col', 'observed_stat', 'simulate_bhbe_null', 'calc_pval']
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
