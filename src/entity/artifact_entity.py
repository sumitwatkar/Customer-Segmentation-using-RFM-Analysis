from collections import namedtuple

# Defining DataIngestionArtifact namedtuple to represent artifacts generated during data ingestion.
DataIngestionArtifact = namedtuple('DataIngestionArtifact',
                                   ['train_file_path','is_ingested','message'])