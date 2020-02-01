### create Cassandra database connection
from dse.cluster import Cluster
from dse.auth import PlainTextAuthProvider

class Connection:
    def __init__(self):
        self.secure_connect_bundle=filepath
        self.path_to_creds=filepath
        self.cluster = Cluster(
            cloud={
                'secure_connect_bundle': self.secure_connect_bundle
            },
            auth_provider=PlainTextAuthProvider(username, password)
        )
        self.session = self.cluster.connect()
    def close(self):
        self.cluster.shutdown()
        self.session.shutdown()