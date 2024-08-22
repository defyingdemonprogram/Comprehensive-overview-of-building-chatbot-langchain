## Graph Databases

Graph databases use graph structures to store and manage data. Instead of organizing data in tables like traditional relational databases, graph databases use:

- **Nodes**: Represent entities such as people, places, or things.
- **Edges**: Represent relationships between nodes, defining how they are connected.
- **Properties**: Store additional attributes or metadata about nodes and edges.

These databases are particularly effective for scenarios where the relationships between data points are crucial. They excel in managing complex, interconnected data, making them ideal for applications such as social networks, recommendation systems, and network management.

### Persistent Volume Setup for Neo4j Database

To ensure data persistence in your Neo4j database, set up a persistent volume with the following command:

```bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    neo4j
```

Access the Neo4j browser at: [http://localhost:7474/browser/](http://localhost:7474/browser/)

**Default Login Credentials:**
- Username: `neo4j`
- Password: `neo4j`

You will need to change the default password before proceeding.

### Changing the Default Password

If you need to change the default password due to an APOC error, follow these steps:

```bash
sudo docker exec -it <container_id> /bin/bash
apt-get update
apt-get install vim
vim /var/lib/neo4j/conf/neo4j.xml
```

### One-Time Instance Setup Without Persistent Data

For a one-time Neo4j instance without data persistence, use:

```bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --env='NEO4JLABS_PLUGINS=["apoc"]' \
    --env='NEO4J_AUTH=neo4j/qwerty12' \
    neo4j
```

Access the Neo4j browser at: [http://localhost:7474/browser/](http://localhost:7474/browser/)


### Restoring the Database from a `.dump` File

1. Copy the `.dump` file to the Neo4j container:

    ```bash
    docker cp DeepLearningAI/data/neo4j.dump <container_id>:/home/
    ```

2. Restore the Neo4j database using the `.dump` file:

    ```bash
    docker exec -it <container_id> bin/neo4j-admin database load --from-path=/home/neo4j.dump --verbose=True --overwrite-destination=True
    ```

### Managing a Movie Database with Neo4j and Python

To manage a movie database using Neo4j with Python, follow these steps:

```python
from langchain_community.graphs import Neo4jGraph

# Initialize Neo4jGraph
graph = Neo4jGraph()

# Import movie data
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id: row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director IN split(row.director, '|') | 
    MERGE (p:Person {name: trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor IN split(row.actors, '|') | 
    MERGE (p:Person {name: trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre IN split(row.genres, '|') | 
    MERGE (g:Genre {name: trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

# Execute the import query
graph.query(movies_query)
```

### Exporting the Database to JSON

To export the database to JSON, use:

```cypher
MATCH (m)-[r]->(n) RETURN m, LABELS(m), r, n
```

### Creating Nodes and Relationships

Use `MERGE` to avoid duplication and ensure efficient node and relationship creation. For example:

```cypher
MERGE (source:Microservice {name: $source_name, technology: $source_technology})
```

This approach avoids the need for specific merge functions for different nodes.