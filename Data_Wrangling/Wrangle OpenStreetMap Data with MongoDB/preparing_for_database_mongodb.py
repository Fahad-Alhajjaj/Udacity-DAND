# Importing needed libraries
import xml.etree.cElementTree as ET
import re
import codecs
import json
import check_postcodes as cp
import check_streets_types as cst
import pdb

# A regex for finding any unwanted chars in a str
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

# A list of key values for grouping their information into a dict when processing an element.
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

# a list of NY, NJ valid postcodes being processed and gathered by get_valid_postcodes function form check_streets_types.py file
valid_postcodes = cp.get_valid_postcodes()

def inner_tags(element, node):
    '''
    It takes an element and process its data into the node dictionary.
    1-if the element tag is an nd tag (node refrence):
         i- it checks to see if a list of nds already exists or not.
        ii- it stores it in the node dictionary by:
            a- appending the element data into the existing index.
            b- Or storing the element data into a new index.

    2-if the element tag is a member tag:
         i- it process element data and checks if the its member type is a node or way.
        ii- it call add_member function to process the data and store the data into the node dictionary.

    3-if the element tag is not an nd or member tag then:
          i- if the the value data is of a unicode type it is ignored.
         ii- check the key for existnce and process its data.
        iii- if the key is an address its data processed and stored:
            a-if the address value is a street name then it's send to audit_street_type function from check_postcodes for process.
            b-if the address value is a postcode then it's send to get_fixed_postcode function from check_streets_types for process.
         iv- if the key has no colons in it then its value is stored into the node dictionary.
          v- if the key has one or more colons then inner_tags_with_multi_colons function is called to processes the key and value.

    Parameters
    ----------
    element : Element
        an osm element to be processed
    node : dict
        a dictionary of all element information so far before adding element parameter information
    
    Returns
    -------
    node : dict
        a dictionary of all element information after process
    '''
    if element.tag == 'nd':
        if 'node_refs' in node:
            node['node_refs'].append(element.attrib['ref'])
        else:
            node['node_refs'] = [element.attrib['ref']]
    elif element.tag == "member":
        is_node = False
        temp = {}
        for k,v in element.attrib.items():
            if k == 'type' and v == 'node':
                is_node = True
            elif k == 'type' and v == 'way':
                pass
            else:
                temp[k] = v
        if is_node:
            node = add_member(node, 'nodes', temp)
        else:
            node = add_member(node, 'ways', temp)
    else:
        if problemchars.match(element.attrib['k']) == None:
            key = element.attrib['k']
            value = element.attrib['v']
            if not isinstance(value, unicode): # checks if value is of unicode type
                if (key.count(':') == 0) and (key not in node):
                    node[key] = value
                elif (key.count(':') == 0) and (key in node):
                    if isinstance(node[key], str):
                        node[key] = {'v1':node[key], 'v2':value}
                    else:
                        node[key][key] = value
                else:
                    if key[:4] == 'addr' and key.count(':') == 1:
                        if cp.is_postcode(key):
                            value = cp.get_fixed_postcode(value, valid_postcodes)
                        elif cst.is_street_name(key):
                            value = cst.audit_street_type(value)
                        if value != None:
                            if 'address' in node:
                                node['address'][key[5:]] = value
                            else:
                                address = {}
                                address[key[5:]] = value
                                node['address'] = address
                    elif key[:4] != 'addr' and key.count(':') > 0:
                        colon_index = key.index(":")
                        if key[:colon_index] in node:
                            value_list = node[key[:colon_index]]
                            if isinstance(value_list, str):
                                temp_dict = {}
                                temp_dict[key[:colon_index]] = value_list
                                node[key[:colon_index]] = inner_tags_with_multi_colons(key[colon_index+1:], value, temp_dict)
                            else:
                                node[key[:colon_index]] = inner_tags_with_multi_colons(key[colon_index+1:], value, value_list)
                        else:
                            value_list = {}
                            value_list = inner_tags_with_multi_colons(key[colon_index+1:], value, value_list)
                            node[key[:colon_index]] = value_list
    return node

def inner_tags_with_multi_colons(key, value, value_list):
    '''
    A recursive function handling a key value that has more than one colon in it.
    It makes a dictionary of nested dictionaries as many as there are colons.

    Parameters
    ----------
    key : str
        The key data of a ndoe
    value : str
        The value data of a node
    value_list: dict
        a dictionary holding information of a node (k, v)
    
    Returns
    -------
    value_list : dict
        a dictionary of all node information
    '''
    if key.count(':') == 0:
        value_list[key] = value
    else:
        colon_index = key.index(":")
        temp_dict = {}
        temp_dict = inner_tags_with_multi_colons(key[colon_index+1:], value, temp_dict)
        value_list[key] = temp_dict
    return value_list

def add_member(node, node_type, values):
    '''
    It takes a value found in a member tag and adds it to the full node dictionary

    Parameters
    ----------
    node : dict
        a dictionary of all element information so far before adding the value parameter
    node_type : str
        type of member node eaither nodes or ways
    values: dict
        a dictionary holding information of member node (ref, role)
    Returns
    -------
    node : dict
        a dictionary of all element information so far including member node
    '''
    if 'members' in node:
        if node_type in node['members']:
            node['members'][node_type].append(values)
        else:
            node['members'][node_type] = []
            node['members'][node_type].append(values)
    else:
        nodes = []
        nodes.append(values)
        node['members'] = {node_type : nodes}
    return node

def shape_element(element, tags=("node", "way", "relation")):
    '''
    1- Takes an osm file element and process it into a python list
    2- passes a child element to the inner_tags function to process it.

    Parameters
    ----------
    element: Element
        an osm element to be processed
    tags : tuple
        tags that are going to be processed
    
    Returns
    -------
    node : dict
        a dictionary of all element information
        None if element of tag is not in tags parameter
    '''
    node = {}
    created_dict = {}
    pos = []
    if element.tag in tags:
        for k, v in element.attrib.items():
            if not isinstance(v, unicode):
                if k in CREATED:
                    created_dict[k] = v
                elif k == 'lat':
                    pos.insert(0,float(v))
                elif k == 'lon':
                    pos.append(float(v))
                else:
                    node[k] = v
        node['created'] = created_dict
        node['pos'] = pos
        node['node_type'] = element.tag
        for child in element:
            node = inner_tags(child, node)
        return node
    else:
        return None

def process_map(file_in, pretty = False):
    '''
    read the input file and process it by passing each element of the osm file to shape_element function.
    write the returned variable by shape_element function to a json file unless the returned variable is None
    then it skips it.

    Parameters
    ----------
    file_in: str
        name of OSM file
    pretty : bool
        a flag to eaither write indentation to the json file or not
        False by default unless overridden by the function call
    '''
    file_out = "{0}.json".format(file_in)
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")