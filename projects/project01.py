
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    # keys 
    keys = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    grades_dict = {}
    for k in keys:
        grades_dict[k] = []
        
    for column in grades.columns:
        if 'checkpoint' in column:
        # get indexes where checkpoint appears + padding
            beginning = column.index('checkpoint')
            end = beginning + len('checkpoint') + 2
            if column[:end] not in grades_dict['checkpoint']:
                grades_dict['checkpoint'].append(column[:end])

        # check with capital M
        elif 'Midterm' in column:
            if column[:len('Midterm')] not in grades_dict['midterm']:
                grades_dict['midterm'].append(column)

        #check with capital F
        elif 'Final' in column:
            if column[:len('Final')] not in grades_dict['final']:
                grades_dict['final'].append(column)

        elif 'discussion' in column:
            if column[:len('discussion') + 2] not in grades_dict['disc']:
                grades_dict['disc'].append(column)

        # check for other assignmnents
        else:
            for k in ['lab', 'project']:
                if k in column: 
                    if column[:len(k) + 2] not in grades_dict[k]:
                        grades_dict[k].append(column[:len(k) + 2])
                        
    return grades_dict

# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    projects = get_assignment_names(grades)['project'] # getting number of projects + names
    
    # get all columns that include project (checkpts too)
    filtered_cols = [column for column in grades.columns if 'project' in column if not('Late' in column or 'check' in column)]
    filt_tab = grades[filtered_cols]
    
    cumulative = [] 
    
    for project in projects:
        # getting only columns for current project 
        project_cols = [column for column in filt_tab.columns if project in column]
        project_tab = filt_tab[project_cols]
        
        max_cols = [column for column in project_tab.columns if 'Max' in column] # columns for max points 
        max_score = sum(project_tab[max_cols].loc[0]) # max points for current project 
        
        st_grades = project_tab.drop(max_cols, axis=1) # get all student grades for current project 
        total = st_grades.sum(axis=1) # add grades for each student 
        
        percentage = total / max_score # get grade percentage 
        
        if len(cumulative) == 0:
            cumulative = percentage
        else: 
            cumulative += percentage
    
    return cumulative / len(projects)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
        
    def conv_to_secs(time):
        times = time.split(':')
        hours = int(times[0])
        minutes = int(times[1])
        seconds = int(times[2])
        
        return hours*60*60 + minutes*60 + seconds
    
    late_lab_col = [column for column in grades.columns if ('lab' in column and 'Lateness' in column)]
    late_lab_tab = grades[late_lab_col]
    grades_late = late_lab_tab.applymap(conv_to_secs)

    threshold = 15000
    
    def late(x):
        if 0 < x <= threshold:
            return 1
        else:
            return 0
        
    grades_late = grades_late.applymap(late)
    grades_late = grades_late.sum()
    # lab01 - Lateness (H:M:S)  --> lab01
    for column in late_lab_col:
        grades_late = grades_late.rename({column:column[:5]})
    return grades_late


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    
    def truly_late(time_log):
        one_week = 7
        two_weeks = 14
        threshold = 10
        hours_late = int(time_log.split(':')[0])
        
        if hours_late < 10:
            return 1.0
        # one week penalty
        if hours_late < (24 * one_week):
            return 0.9
        # two weeks
        if hours_late < (24 * two_weeks):
            return 0.7
        # beyond two weeks
        if hours_late > (24 * two_weeks):
            return 0.4
    
    return col.apply(truly_late)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    lab_grades = {}
    # lab names
    lab_names = get_assignment_names(grades)['lab'] 
    
    for lab in lab_names:
        current = grades[[column for column in grades.columns if (lab in column)]]
        lab_points = current[lab]
        lab_max = current[lab + ' - Max Points']
        lab_late = current[lab + ' - Lateness (H:M:S)']
        late_penalty = lateness_penalty(lab_late)
        
        # change to dataframe
        lab_grades[lab] = ((lab_points / lab_max) * late_penalty).tolist()
    
    return pd.DataFrame(lab_grades)


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    processed = processed.fillna(0)
    
    # calculate total of each student by row 
    def get_total(row):
        # reps all grades
        grades = row.values
        # drop lowest 
        grades = np.delete(grades, grades.argmin()) 
        return grades.mean() 
    
    return processed.apply(get_total, axis = 1)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
        
    #lab, proj, checkpt, exams (mt, final)
    grades = grades.fillna(value = 0)
    # use prev funcs to calculate grades
    proccessed_labs = process_labs(grades)
    lab_grades = lab_total(proccessed_labs)
    project_grades = projects_total(grades)
    assignment_dict = get_assignment_names(grades)
    columns = list(grades.columns)
    keys = ['midterm', 'final', 'disc', 'checkpoint']
    final_grades = pd.DataFrame()
    
    # loop through assignments
    for key in keys:
        df = pd.DataFrame()
        assignments = assignment_dict[key]
        for assignment in assignments:
            max_col_nm = ''
            for col in columns:
                if (assignment in col) & ('Max' in col):
                    max_col_nm = col
            df[assignment] = grades[assignment] / grades[max_col_nm]
        final_grades[key] = df.mean(axis = 1)
    
    # calc group scores
    final_grades['lab'] = lab_grades * .2
    final_grades['projects'] = project_grades * .3
    final_grades['checkpoint'] = final_grades['checkpoint'] *.025
    final_grades['disc'] = final_grades['disc'] *.025
    final_grades['midterm'] = final_grades['midterm'] *.15
    final_grades['final'] = final_grades['final'] *.3
    
    return final_grades.sum(axis = 1)


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def determine_letter(grade):
        if grade >= .9:
            return "A"
        if grade >= .8:
            return "B"
        if grade >= .7:
            return "C"
        if grade >= .6:
            return "D"
        else:
            return "F"

    return total.apply(determine_letter)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    
    total = total_points(grades)
    letter_grades = final_grades(total)

    return letter_grades.value_counts() / len(letter_grades)
# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------
def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    by_year = grades['Level']
    grades_student = pd.Series(total_points(grades), name = "Total")
    grade_lev_df = pd.concat([by_year, grades_student], axis = 1)
    
    # new dataframe of seniors
    seniors = (grade_lev_df['Level'] == 'SR').sum()
    # avg of observed
    obs_avg = grade_lev_df[grade_lev_df["Level"] == "SR"].get("Total").mean()

    averages = []
    # N simulations
    # numpy arrays are faster 
    for _ in np.arange(N):
        random_sample = grade_lev_df['Total'].sample(seniors, replace = False)
        avg = np.mean(random_sample)
        averages.append(avg)
    
    # p-value
    return (pd.Series(averages) <= obs_avg).mean()

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    grades = grades.fillna(value=0)
    processed_labs = process_labs(grades)
    num_rows = processed_labs.shape[0]
    num_cols = processed_labs.shape[1]
    noise = np.random.normal(0, 0.02, size=(num_rows, num_cols))
    
    # lab w/ noise
    processed_labs = processed_labs + noise
    for col in processed_labs.columns:
        processed_labs[col] = np.clip(processed_labs[col], 0, 1)
    lab_grades = lab_total(processed_labs)
    
    grades = grades.fillna(value=0)
    columns = list(grades.columns)

    # cols are free responses and max are maxes 
    # proj w/ noise
    cols = []
    cols_max = []
    free_cols = []
    free_cols_max = []
    for col in columns:
        if 'project' in col and 'checkpoint' not in col and 'Lateness' not in col:
            if "free" in col:
                if 'Max' in col:
                    free_cols_max.append(col)
                else:
                    free_cols.append(col)
            else: 
                if 'Max' in col:
                    cols_max.append(col)
                else:
                    cols.append(col)
    proj_df = pd.DataFrame()
    
    for col in cols:
        max_col_nm =''
        free_col_nm =''
        max_free_col_nm =''
        for max_col in cols_max:
            if col in max_col:
                max_col_nm = max_col
        for free_col in free_cols:
            if col in free_col:
                free_col_nm = free_col
        for max_free_col in free_cols_max:
            if col in max_free_col:
                max_free_col_nm = max_free_col
        if free_col_nm == '':
            proj_df[col] = grades[col] / grades[max_col_nm]
        else:
            proj_df[col] = (grades[col] + grades[free_col_nm]) / (grades[max_col_nm] + grades[max_free_col_nm])
    num_rows = proj_df.shape[0]
    num_cols = proj_df.shape[1]
    noise = np.random.normal(0, 0.02, size = (num_rows, num_cols))
    proj_df = proj_df + noise
    for col in proj_df.columns:
        proj_df[col] = np.clip(proj_df[col], 0, 1)
    project_grades = proj_df.mean(axis = 1)
    
    # all else
    assignment_dict = get_assignment_names(grades)
    columns = list(grades.columns)
    norm_assignments = ['midterm', 'final', 'disc', 'checkpoint']
    output = pd.DataFrame()
    for key in norm_assignments:
        df = pd.DataFrame()
        assignment_names = assignment_dict[key]
        for assignment in assignment_names:
            max_col_nm = ''
            for col in columns:
                if (assignment in col) & ('Max' in col):
                    max_col_nm = col
            df[assignment] = grades[assignment] / grades[max_col_nm]
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        noise = np.random.normal(0, 0.02, size=(num_rows, num_cols))
        df = df + noise
        for col in df.columns:
            df[col] = np.clip(df[col], 0, 1)    
        output[key] = df.mean(axis = 1)
        
    # sum of scores     
    output['lab'] = lab_grades * .2
    output['projects'] = project_grades * .3
    output['midterm'] = output['midterm'] *.15
    output['final'] = output['final'] *.3
    output['disc'] = output['disc'] *.025
    output['checkpoint'] = output['checkpoint'] *.025
    
    
    return output.sum(axis = 1)


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return [0.0009719330272155435, 82.80, [80.08, 86.17], 0.0654205, [True, True]]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
