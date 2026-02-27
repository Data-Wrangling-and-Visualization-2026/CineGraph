from db.repositories.graph_repo import GraphRepository
from db.session import SessionLocal


class GraphService:
    def __init__(self):
        self.db = SessionLocal()
        self.repo = GraphRepository(self.db)

    def construct_graph(self):
        pass