CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE graph (
    id SERIAL PRIMARY KEY,
    path ltree,
    name VARCHAR(100),
    type VARCHAR(10),
    children_count INT
);

CREATE INDEX path_gist_idx ON graph USING GIST (path);
CREATE INDEX path_idx ON graph USING btree (path);

CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100),
    year INT,
    other_data JSONB,
    graph_id INT REFERENCES graph(id)
);

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    movie_id INT REFERENCES movies(id),
    window_id INT,
    embedding VECTOR(6)
);
