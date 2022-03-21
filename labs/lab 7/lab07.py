import os
import pandas as pd
import numpy as np
import requests
import json
import re



# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


####################
#  Regex
####################



# ---------------------------------------------------------------------
# Problem 1
# ---------------------------------------------------------------------

# A string that has a [ as the third character and ] as the sixth character.
def match_1(string):
    """
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    #Your Code Here
    # dot takes place of 1st, 2nd character, then "[" is third
    # two more dots for 4th & 5th, then 6th --> \ bcus brackets LOL
    pattern = '^..\[..\]'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_2(string):
    """
    Phone numbers that start with '(858)' and
    follow the format '(xxx) xxx-xxxx' (x represents a digit)
    Notice: There is a space between (xxx) and xxx-xxxx

    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    #Your Code Here
    # d{3 dig} 
    pattern = '^\((858)\) \d{3}-\d{4}$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_3(string):
    """
    Find a pattern whose length is between 6 to 10
    and contains only word character, white space and ?.
    This string must have ? as its last character.

    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    #Your Code Here

    pattern = '^[a-zA-Z ?]{5,9}\?$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    A string that begins with '$' and with another '$' within, where:
        - Characters between the two '$' can be anything except the 
        letters 'a', 'b', 'c' (lower case).
        - Characters after the second '$' can only have any number 
        of the letters 'a', 'b', 'c' (upper or lower case), with every 
        'a' before every 'b', and every 'b' before every 'c'.
            - E.g. 'AaBbbC' works, 'ACB' doesn't.

    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False

    >>> match_4("$iiuABc")
    False
    >>> match_4("123$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    """
    #Your Code Here
    pattern = '^\$[^a-c]*\$[aA]+[bB]+[cC]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    A string that represents a valid Python file name including the extension.
    *Notice*: For simplicity, assume that the file name contains only letters, numbers and an underscore `_`.

    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """

    #Your Code Here
    pattern = '\w+\.py$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    Find patterns of lowercase letters joined with an underscore.
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """

    #Your Code Here
    pattern = '^[a-z]+_[a-z]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_7(string):
    """
    Find patterns that start with and end with a _
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """

    pattern = '^_.*_$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_8(string):
    """
    Apple registration numbers and Apple hardware product serial numbers
    might have the number "0" (zero), but never the letter "O".
    Serial numbers don't have the number "1" (one) or the letter "i".

    Write a line of regex expression that checks
    if the given Serial number belongs to a genuine Apple product.

    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """

    pattern = '^[^OiI1]*$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



# Check if a given ID number is from Los Angeles (LAX),
# San Diego(SAN) or the state of New York (NY).
# ID numbers have the following format SC-NN-CCC-NNNN.
def match_9(string):
    '''
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    '''


    pattern = '^(NY)-\d{2}-(NYC)-\d{4}$|^(CA)-\d{2}-(SAN)-\d{4}$|^(CA)-\d{2}-(LAX)-\d{4}$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    Given an input string, cast it to lower case, remove spaces/punctuation, 
    and return a list of every 3-character substring that satisfy the following:
        - The first character doesn't start with 'a' or 'A'
        - The last substring (and only the last substring) can be shorter than 
        3 characters, depending on the length of the input string.
    
    >>> match_10('ABCdef')
    ['def']
    >>> match_10(' DEFaabc !g ')
    ['def', 'cg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10( "Ab..DEF")
    ['def']

    '''

    low_and_replaced = re.sub(' ', '', string.lower())
    substrings = re.findall('\S{1,3}', low_and_replaced)
    new_string = ''
    for substring in substrings:
        search_output = re.search('^[^aA]{1}\S*$', substring)
        if search_output != None:
            no_punctuation = re.sub('[^\w]','', search_output[0])
            new_string+=no_punctuation
    output = re.findall('\S{1,3}', new_string)
    return output



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_personal(s):
    """
    :Example:
    >>> fp = os.path.join('data', 'messy.test.txt')
    >>> s = open(fp, encoding='utf8').read()
    >>> emails, ssn, bitcoin, addresses = extract_personal(s)
    >>> emails[0] == 'test@test.com'
    True
    >>> ssn[0] == '423-00-9575'
    True
    >>> bitcoin[0] == '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2'
    True
    >>> addresses[0] == '530 High Street'
    True
    """
    
    def remove_pre(strings, pre):
        removal_helper = lambda x : x.replace(pre,'')
        return list(map(removal_helper,strings))
    # email patt
    email_pattern = '[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]{3}'
    emails = re.findall(email_pattern,s)
    # social patt
    ssn_pattern = 'ssn:\d{3}-\d{2}-\d{4}'
    ssn = re.findall(ssn_pattern, s)
    ssn_fixed = remove_pre(ssn,'ssn:')
    fil1 = lambda x : x[:3] not in ['000','666']
    fil2 = lambda x : x[:1] != '9'
    ssn_complete = list(filter(fil2,list(filter(fil1,ssn_fixed))))
    # bitcoin addy
    bit_pattern = 'bitcoin:[a-zA-Z0-9]{5,}'
    bit = re.findall(bit_pattern,s)
    bitcoin_addy = remove_pre(bit,'bitcoin:')
    # street address
    addy_pattern = '\d+ [a-zA-z]+ [a-zA-z]+'
    addies = re.findall(addy_pattern, s)
    return tuple([emails, ssn_complete, bitcoin_addy, addies])

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def tfidf_data(review, reviews):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> out['cnt'].sum()
    85
    >>> 'before' in out.index
    True
    """
    df = pd.DataFrame(pd.DataFrame({'words':re.findall('\S+',review)}).groupby('words').size()).rename(columns = {0:'cnt'})
    # term freq
    df['tf'] = df['cnt'] / df['cnt'].sum()
    idf_list = []
    for word in df.index:
        pattern = '\\b%s\\b' % word
        idf = np.log((len(reviews)/reviews.str.contains(pattern).sum()))
        idf_list.append(idf)
    df['idf'] = idf_list
    df['tfidf'] = df['tf'] * df['idf']
    return df


def relevant_word(out):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> relevant_word(out) in out.index
    True
    """
    return out['tfidf'].idxmax()


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def hashtag_list(tweet_text):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = hashtag_list(test['text'])
    >>> (out.iloc[0] == ['NLP', 'NLP1', 'NLP1'])
    True
    """

    pattern = '#\S+'
    output = []
    # helper to retrieve hashtags in the tweet text
    def get_hashtags(string):
        ht_list = pd.Series(re.findall(pattern,string))
        # return as a list
        return list(map(lambda x : x.replace('#', ''), ht_list))
    output = tweet_text.apply(get_hashtags)
    return output


def most_common_hashtag(tweet_lists):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = hashtag_list(pd.DataFrame(testdata, columns=['text'])['text'])
    >>> most_common_hashtag(test).iloc[0]
    'NLP1'
    """

    total = pd.Series(tweet_lists.sum())
    count_df = total.value_counts()
    
    # helper function to change entry depending on # of hashtags
    def helper(hashtags):
        # no hashtags 
        if len(hashtags) == 0:
            return np.nan
        # one distinct
        elif len(hashtags) == 1:
            return hashtags[0]
        # more than 1
        else:
            return count_df.loc[hashtags].idxmax()
    return tweet_lists.apply(helper)


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------


def create_features(ira):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = create_features(test)
    >>> anscols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    >>> ansdata = [['text cleaning is cool', 3, 'NLP1', 1, 1, True]]
    >>> ans = pd.DataFrame(ansdata, columns=anscols)
    >>> (out == ans).all().all()
    True
    """
    df = pd.DataFrame()
    df['text'] = ira['text'].apply(lambda x : clean(x))
    # number of hashtags present in a tweet
    df['num_hashtags'] = pd.Series([len(x) for x in hashtag_list(ira['text'])])
    # gives the most common hashtag associated to a tweet
    df['mc_hashtags'] = pd.Series(most_common_hashtag(hashtag_list(ira['text'])))
    # number of tags a given tweet has (look for the presence of @)
    df['num_tags'] = ira['text'].apply(lambda x : len(re.findall('@[A-Za-z0-9]+', x)))
    # number of hyper-links present in a given tweet, http
    df['num_links'] = ira['text'].apply(lambda x : len(re.findall('http.*', x)))
    # retweets
    df['is_retweet'] = ira['text'].apply(lambda x : is_retweet(x))
    return df

def is_retweet(string):
    pattern = 'RT'
    if re.search(pattern, string) == None:
        return False
    else:
        return True

def clean(string):
    """
    helper function to clean the text according to:
    - The non-alphanumeric characters removed (except spaces),
    - All words should be separated by exactly one space,
    -  The characters all lowercase,
    - All the meta-information above (Retweet info, tags, hyperlinks, hashtags) removed.
    """
    string = re.sub('@[A-Za-z0-9]+', '', string)
    string = re.sub('#[A-Za-z0-9]+', '', string)
    string = re.sub('RT', '', string)
    string = re.sub('http.*', '', string)
    string = re.sub('[^A-Za-z0-9 ]',' ', string)
    # all are lowercase
    string = string.lower().lstrip().rstrip()
    return string

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['match_%d' % x for x in range(1, 10)],
    'q02': ['extract_personal'],
    'q03': ['tfidf_data', 'relevant_word'],
    'q04': ['hashtag_list', 'most_common_hashtag'],
    'q05': ['create_features']
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
