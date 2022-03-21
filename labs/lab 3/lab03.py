
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    
    # are these tires faulty
    
    return [3, 6, 7]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    # at most 60% limits the faultiness
    return [2, 5, 8]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    # 3: "number of attempts"
    return [1, 2, 4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 5


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    '''
    table = df.copy()
    # keep reviews as int
    table["Reviews"] = df["Reviews"].astype(int)
    # strip letters from size column, conv to float
    def strip_size(chars):
        return chars.strip('M').strip('k')
    table["Size"] = table["Size"].apply(strip_size)
    table["Size"] = table["Size"].astype(float)
    # strip "+" from installs, remove commas, conv to integer
    def strip_installs(chars):
        return int(chars.strip("+").replace(",", ""))
    table["Installs"] = table["Installs"].apply(strip_installs)
    # change all the 'Free's to 1 and the 'Paid's to 0
    def clean_type(x):
        if x == "Free":
            return 1
        else:
            return 0
    table["Type"] = table["Type"].apply(clean_type)
    # Strip dollar sign
    # convert it to correct numeric data type
    def strip_price(price):
        return float(price.strip("$"))
    table["Price"] = table["Price"].apply(strip_price)
    # Strip all but the year (e.g. 2018) from Last Updated
    # and convert it to type int
    def strip_last_updated(date):
        return int(date[-4:])
    table["Last Updated"] = table["Last Updated"].apply(strip_last_updated)

    return table

def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    # find the year with the highest median Installs
    # among all years with at least 100 apps
    num_years = cleaned.groupby("Last Updated").count()
    min_100_apps = num_years[num_years["App"] >= 100].index
    filtered_table = cleaned[cleaned["Last Updated"].isin(min_100_apps)]
    median_installs = filtered_table[['Last Updated','Installs']].groupby('Last Updated').median()
    highest_med = median_installs.idxmax()[0]
    
    # Find the Content Rating with the highest minimum Rating
    cont_rating_mins = cleaned[['Content Rating', 'Rating']].groupby('Content Rating').min()
    highest_min_rating = cont_rating_mins.idxmax()[0]
    
    # Find the Category has the highest average price
    avg_prices_cat = cleaned[['Category','Price']].groupby('Category').mean()
    highest_avg_price = avg_prices_cat.idxmax()[0]

    # Find the Category with lowest average rating,
    # among apps that have at least 1000 review
    min_1000_reviews = cleaned[cleaned["Reviews"] >= 1000]
    avg_rat_cat = min_1000_reviews[['Category', 'Rating']].groupby('Category').mean()
    lowest_avg_rat = avg_rat_cat.idxmin()[0]

    return [highest_med, highest_min_rating, highest_avg_price, lowest_avg_rat]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """

    copy = cleaned.copy()
    mean = cleaned[['Category','Reviews']].groupby('Category').mean()
    standard_dev = cleaned[['Category','Reviews']].groupby('Category').std()
    
    def standardize_reviews(x):
        return np.array([x[0],(x[1] - mean.loc[x[0],:]['Reviews']) / (standard_dev.loc[x[0],:]['Reviews'])])
    copy["Reviews"] = cleaned[['Category','Reviews']].transform(standardize_reviews, axis = 1)
    return copy[["Category", "Reviews"]]


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ["equal", "GAME"]


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    directory = os.listdir(dirname)
    read_files = []
    for file in directory:
        df = pd.read_csv(dirname + '/' + file)
        cols = []
        for col in df.columns:
            cols.append(col.replace('_', ' ').lower())
        df.columns = cols
        read_files.append(df)
              
    return pd.concat(read_files)


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """

    highest_num_empl = max(df.groupby('current company').size())
    # emails that end in .edu
    def check_edu(email):
        temp = str(email)
        if temp[-4:] == '.edu':
            return True
        else:
            return False
    edu_emails = len(list(filter(check_edu, df['email'])))
    base = 0
    longest_name = ''
    num_mana = 0
    # longest job name
    for title in df['job title']:
        temp = len(str(title))
        if temp > base:
            base = temp
            longest_name = str(title)
        fixed = str(title).lower()
        # num managers
        if 'manager' in fixed:
            num_mana += 1
    
    return [highest_num_empl, edu_emails, longest_name, num_mana]

# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    directory = os.listdir(dirname)
    tables = []
    for file in directory:
        df = pd.read_csv(dirname + "/" + file)  
        tables.append(df.set_index('id'))
    return pd.concat(tables, sort = True, axis = 1)


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    # helper to get student's ec pts earned
    def ec_pts(student):
        incomplete_surveys = student.isnull().sum()
        if incomplete_surveys > 1:
            return 0 
        else:
            return 5
    # at least one q
    def check_class(df):
        bool_ar = (df.isnull().sum().values / df.shape[0]) <= .10
        if True in bool_ar:
            return True
        else:
            return False
    df['extra credit'] = df.apply(ec_pts, axis = 1)
    # everyone receives one point if 90% class answers at least one q
    if check_class(df) == True:
        df['extra credit'] = df['extra credit'] + 1
        
    return df[['name', 'extra credit']]

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    What is the most popular Procedure Type for all of the pets we have in our `pets` dataset?
​
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out,str)
    True
    """

  # match procedure w pet
    combined_df = pets.merge(procedure_history, how = "inner", on = "PetID")
    return combined_df.groupby("ProcedureType").count().idxmax()[0]
   

def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """

  # What is the name of each customer's pet(s)? 
    # takes in owners, pets 
    # returns a Series that holds the pet name (as a string)
    # indexed by owner's (first) name.
    # If an owner has multiple pets,
    # the corresponding value should be a list of names as strings.

    def pet_num(pets):
        pets = list(pets)
        if len(pets) == 1:
            return pets[0]
        else:
            return pets
    owners_ID_pets = pets.groupby('OwnerID')['Name'].apply(pet_num)
    merged_df = owners.merge(owners_ID_pets.to_frame(), how = 'inner', on = 'OwnerID')[['Name_x','Name_y']]
    return merged_df.set_index(['Name_x'])['Name_y']


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
​
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    
    merged_df = pets.merge(owners, how = 'inner', on = 'OwnerID').merge(procedure_history, how = 'inner', on = 'PetID').merge(procedure_detail, how = 'inner', on = 'ProcedureType')
    return merged_df.groupby('City').sum()['Price']



# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['most_popular_procedure', 'pet_name_by_owner', 'total_cost_per_city']
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
