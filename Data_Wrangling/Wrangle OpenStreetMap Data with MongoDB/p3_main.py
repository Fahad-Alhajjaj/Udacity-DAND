#Importing needed libraries
import preparing_for_database_mongodb as pmongo

OSMFILE = "brooklyn_new-york.osm"

def run():
	'''
	First function to be called and it initiate the osm file process.
	by calling process_map function from preparing_for_database_mongodb file.
	'''
	pmongo.process_map(OSMFILE, True)

if __name__ == "__main__":
    run()