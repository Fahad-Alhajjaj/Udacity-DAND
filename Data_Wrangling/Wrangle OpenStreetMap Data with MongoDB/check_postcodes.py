'''
zip_code_database.csv file has been obtained from https://www.unitedstateszipcodes.org/ny/#zips-list
'''

# Importing needed libraries and files
import pandas

def add_leading_zeros(zip_code):
    '''
    If input Parameter length is less than 5, add leading zeros to make it a five digit zipcode str

    Parameters
    ----------
    zip_code : str
        less than 5 chars str (zip code)

    Returns
    -------
    zip_code : str
        a 5 chars str (zip code)
    '''
    while len(zip_code) < 5:
        zip_code = '0' + zip_code
    return zip_code

def get_fixed_postcode(value, valid_zipcodes):
    '''
    gets a zip code (value) and check it with the valid list of zip codes (valid_zipcodes)
    if it does not valid, try and fix it by deleting unneeded chars then check it again.
    if still not valid then None is returned

    Parameters
    ----------
    value: str
        a zip code str
    valid_zipcodes : list
        a list of valid zip codes str
    
    Returns
    -------
    zip_code : str, None
        a valid zip code or None
    '''
    valid = None
    while True:
        if value not in valid_zipcodes:
            if value[0] == 'N':
                valid = value[-5:]
            elif (len(value) > 5 and len(value) <= 10) or (';' in value):
                valid = value[:5]
            else:
                valid = None
                break
            value = valid
        else:
            valid = value
            break 
    return valid

def get_valid_postcodes():
    '''
    read in 'zip_code_database.csv' file and extracts the valid list of zip codes for NY and NJ states
    
    Returns
    -------
    zip_codes : list
        a valid list of zip codes for NY and NJ states
    '''
    zip_codes = []
    postcode_df = pandas.read_csv('zip_code_database.csv')
    postcode_df = postcode_df[(postcode_df['state'] == 'NY') | (postcode_df['state'] == 'NJ')]
    temp_list = postcode_df['zip'].tolist()
    for item in temp_list:
        zip_code = str(item)
        if len(zip_code) < 5:
            zip_code = add_leading_zeros(zip_code)
            zip_codes.append(zip_code)
        else:
            zip_codes.append(zip_code)
    return zip_codes

def is_postcode(key):
    '''
    checks if the input Parameter is equale to the str "addr:postcode"

    Parameters
    ----------
    key: str
        a string key for an element attribute
    
    Returns
    -------
    : bool
        True:  if the Parameter is equale to "addr:postcode"
        False: otherwise 
    '''
    return (key == "addr:postcode")