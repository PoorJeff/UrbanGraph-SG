# Neo4j Community Edition 5.x
# Used as the graph database backend for UrbanGraph-SG
FROM neo4j:5-community

# Copy initialization scripts (optional)
# COPY docker/neo4j-init.cypher /docker-entrypoint-initdb.d/

# Default environment variables (override in docker-compose.yml or .env)
ENV NEO4J_AUTH=neo4j/urbangraph-sg-dev
ENV NEO4J_server_memory_heap_initial__size=1G
ENV NEO4J_server_memory_heap_max__size=2G
ENV NEO4J_server_memory_pagecache_size=500M

# APOC plugin for utility procedures
ENV NEO4J_PLUGINS='["apoc"]'
