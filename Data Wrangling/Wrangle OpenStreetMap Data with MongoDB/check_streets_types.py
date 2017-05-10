# Importing needed libraries and files
import re

# a regex variable that gets the last word in a str variable.
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


# expected street types.
expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Walk", "Terrace", "Slip", "Alley", "Crescent", "Highway",
            "Expressway", "Extension", "Loop", "Plaza", "East", "Southeast", "South", "Southwest", "West",
            "Northwest", "North", "Northeast", "Broadway", "Finest", "Americas"]

# expected street names
streets = [ "Avenue A", "Avenue B", "Avenue C", "Avenue D", "Avenue E", "Avenue F", "Avenue G", "Avenue H", "Avenue I",
            "Avenue J", "Avenue K", "Avenue L", "Avenue M", "Avenue N", "Avenue O", "Avenue P", "Avenue Q", "Avenue R",
            "Avenue S", "Avenue T", "Avenue U", "Avenue V", "Avenue W", "Avenue X", "Avenue Y", "Avenue Z", "Washington Square Village",
            "Avenue Of The Americas", "Avenue of Puerto Rico", "Union Turnpike", "Colonial Gardens", "Kings Highway"]

# mapping found types to expected types. 
mapping = { "St" : "Street",
            "St." : "Street",
            "St.,": "Street",
            "St," : "Street",
            "st" : "Street",
            "ST" : "Street",
            "street" : "Street",
            "Streeet": "Street",
            "Steet" : "Street",
            "Streeet" : "Street",
            "avenue" : "Avenue",
            "Ave," : "Avenue",
            "ave" : "Avenue",
            "Ave" : "Avenue",
            "Ave." : "Avenue",
            "Avene" : "Avenue",
            "Avenue," : "Avenue",
            "AVE." : "Avenue",
            "Blvd" : "Boulevard",
            "Rd." : "Road",
            "Rd" : "Road",
            "Dr" : "Drive",
            }


def edit_name(street_name):
    '''
    takes a street name and fix any street types in the name if there is any using the mapping variable.

    Parameters
    ----------
    street_name : str
        a full street name

    Returns
    -------
    street_name : str
        a fixed street
    '''
    name_list = street_name.split(" ")
    i=0
    while i<len(name_list):
        word = name_list[i]
        if (word in mapping) and (word not in expected):
            street_name = street_name.replace(word, mapping[word])
            name_list = street_name.split(" ")
        i+=1
    return street_name

def audit_street_type(street_name):
    '''
    takes a street name and call edit_name function to fix any problems with street type
    and then check if the street name is an excepted name it return it
    if not the function will check if an expected type exists in the street name and discard any
    information after it
    if the street name can't be fixed and is not exceptable it will return None

    Parameters
    ----------
    street_name : str
        a full street name

    Returns
    -------
    street_name : str
        an excepted and valid street name, or None if it is not.
    '''
    street_name = edit_name(street_name)
    changed = False
    a = street_type_re.search(street_name)
    if a:
        street_type = a.group()
        if (street_type not in expected) and (street_name not in streets):
            for name in expected:
                if name in street_name:
                    index = street_name.find(name)
                    if index >= 0:
                        index+= len(name)
                        street_name = street_name[:index]
                        changed = True
            if not changed:
                return None
    return street_name


def is_street_name(key):
    '''
    checks if the input Parameter (key) is equale to the str "addr:street"

    Parameters
    ----------
    key: str
        a string key for an element attribute
    
    Returns
    -------
    : bool
        True:  if the Parameter is equale to "addr:street"
        False: otherwise 
    '''
    return (key == "addr:street")