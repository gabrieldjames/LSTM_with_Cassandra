### create Cassandra database connection
from dse.cluster import Cluster
from dse.auth import PlainTextAuthProvider

class Connection:
    def __init__(self):
        self.secure_connect_bundle='C:\\Users\\gabej\\Documents\\MSBA\\Independent_Proj\\Cassandra_Proj\\creds.zip'
        self.path_to_creds=''
        self.cluster = Cluster(
            cloud={
                'secure_connect_bundle': self.secure_connect_bundle
            },
            auth_provider=PlainTextAuthProvider('ssstutter', 'Bigfalls2!')
        )
        self.session = self.cluster.connect()
    def close(self):
        self.cluster.shutdown()
        self.session.shutdown()
