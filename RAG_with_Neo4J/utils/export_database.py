# This script extract the nodes and relation form the Neo4J database
# and save in json format
# 
# Here label is stored as the label key in property of node. 
# If there is a node in your databse with property label, change the Label key to something unique
#  
# Author: @realsanjeev
import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Load Neo4j connection parameters from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define the Cypher query to relationship object
# relationship object conatains start node, end node and its relationship
query_for_node_with_relation = """
MATCH (n)-[r]->(m)
RETURN n, r, m
"""

# query to get node without relationship
query_for_node_without_relation = """
MATCH (n)
WHERE NOT (n)--()
RETURN n
"""

# define query to get 
# Execute the Cypher query
with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run(query_for_node_with_relation)

    # Process the query results
    export_data = []
    for record in result:
        # get relationship object
        relationship_obj = record['r']
        # we are adding label propert to node
        # label can be multiple and is frozenset obj, 
        # so we store in list to make it json serializable
        start_node_obj = relationship_obj.start_node

        # Get node properties except "textEmbedding"
        start_node_data = {k: v for k, v in start_node_obj.items() if k != "textEmbedding"}
        start_node = dict(start_node_data, 
                          **{"label": list(start_node_obj.labels), 
                             "id": int(start_node_obj.element_id.split(":")[-1])})
        
        relationship_data = {
            "type": relationship_obj.type,
            "property": dict(relationship_obj.items())
        }
        
        end_node_obj = relationship_obj.end_node
        end_node_data = {k: v for k, v in end_node_obj.items() if k != "textEmbedding"}

        end_node = dict(end_node_obj, 
                        **{"label": list(end_node_obj.labels), 
                            "id": int(end_node_obj.element_id.split(":")[-1])})


        # Add node, relationship, and connected node data to export_data list
        export_data.append({
            'start_node': start_node,
            'relationship': relationship_data,
            'end_node': end_node
        })
        # print(export_data)
        # break


with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run(query_for_node_without_relation)

    for record in result:
        # node without relationship
        lonely_node = dict(record["n"], **{"label": list(record["n"].labels)})

        # Add node to export_data list
        export_data.append({
            'lonely_node': start_node,
        })

# Save the exported data to a JSON file
output_file = "exported_data.json"
with open(output_file, 'w') as f:
    json.dump(export_data, f, indent=4)

# Close the Neo4j driver
driver.close()
