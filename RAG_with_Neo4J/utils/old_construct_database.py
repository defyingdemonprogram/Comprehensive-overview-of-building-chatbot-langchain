import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

def make_valid_query(query: str) -> str:
    return query.replace("{'", "{").replace("':", ":").replace(", '", ", ")

# Load data from JSON file
JSON_PATH = "/home/sanjeev/Desktop/Trainee/trainee/RAG/myWork/DeepLearningAI/data/exported_data.json"
with open(JSON_PATH, 'r') as f:
    export_data = json.load(f)

# Connect to the new Neo4j database
new_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define the query template for creating relationship nodes
QUERY_CREATE_RELATION_NODE = """
MERGE (source:{start_node_label} {start_node_props})-[r:{relationship_type}]->(target:{target_node_label} {target_node_props})
// WHERE ID(source) = $source_id AND ID(target) = $target_id
SET r += {relationship_props}
"""
QUERY_CREATE_RELATION_NODE = """
MATCH (source:{start_node_label} {start_node_props}), (target:{target_node_label} {target_node_props})
WHERE source.id = {source_id} AND source.id = {target_id}
CREATE (source)-[r:{relationship_type}]->(target)
SET r += {relationship_props}
"""

QUERY_CREATE_RELATION_NODE = """
MATCH (source:{start_node_label}), (target:{target_node_label})
WHERE source.id = {source_id} AND target.id = {target_id}
MERGE (source)-[r:{relationship_type}]->(target)
SET r += {relationship_props}
"""

# Iterate through the exported data
for record in export_data:
    try:
        if "lonely_node" in record:
            node_data = record["lonely_node"]
            # Pop label name from node data
            label = node_data.pop("label")

            with new_driver.session() as session:
                match_query = f"MERGE (n:{label[0]} {node_data}) RETURN n"
                session.run(query, props=node_data)
            continue

        # Retrieve start and end nodes along with relationship data
        start_node = record['start_node']
        start_node_label = start_node.pop('label')
        relationship_data = record['relationship']
        end_node = record['end_node']
        target_node_label = end_node.pop('label')

        # Write node with relationship
        with new_driver.session() as session:
            # first create start node
            first_query = f"MERGE (n:{start_node_label[0]} {start_node}) RETURN n"
            # first_query = make_valid_query(first_query)
            session.run(first_query)

            # create last node
            end_node_query = f"MERGE (n:{target_node_label[0]} {end_node}) RETURN n"
            # end_node_query = make_valid_query(end_node_query)
            session.run(end_node_query)

            # get source id and target id
            source_id = start_node.pop("id")
            target_id = end_node.pop('id')

            # # test the validaity
            # query = f"""
            #     MATCH (source:{start_node_label[0]}), (target:{target_node_label[0]})
            #     WHERE source.id = {source_id} AND target.id = {target_id}
            #     return source, target
            #     """.format()
            # query = make_valid_query(query=query)
            # print(query)
            # reuslt = session.run(query)
            # # print result
            # print(reuslt.single())
            # break

            query = QUERY_CREATE_RELATION_NODE.format(
                start_node_label=start_node_label[0],
                source_id=source_id,
                relationship_type=relationship_data["type"], 
                target_node_label=target_node_label[0],
                # start_node_props=start_node,
                relationship_props=relationship_data["property"],
                # target_node_props=end_node,
                target_id=target_id
            )
            query = make_valid_query(query=query)
            session.run(query)

    except Exception as err:
        print(f"[ERROR]: {err}")
        print("-" * 100)
        break

# Close the new driver
new_driver.close()
