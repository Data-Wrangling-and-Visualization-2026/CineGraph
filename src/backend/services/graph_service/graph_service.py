from db.repositories.graph_repo import GraphRepository
from db.session import SessionLocal


class GraphService:
    """
    The services for graph related operations excluding creation
    """
    def __init__(self):
        self.db = SessionLocal()
        self.repo = GraphRepository(self.db)

    def construct_graph(self):
        """
        All the creation logic is at './clustering'
        """
        pass