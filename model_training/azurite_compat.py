"""Small Azure Blob adapter exposing the legacy object-store call shape."""

from types import SimpleNamespace

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient


class AzuriteBlobCompat:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        del secure
        self.client = BlobServiceClient(
            account_url=endpoint,
            credential={"account_name": access_key, "account_key": secret_key},
        )

    def _container(self, name: str):
        container = self.client.get_container_client(str(name))
        try:
            container.create_container()
        except ResourceExistsError:
            pass
        return container

    def bucket_exists(self, bucket_name: str) -> bool:
        return self._container(bucket_name).exists()

    def list_objects(self, bucket_name: str, prefix: str, recursive: bool = True):
        del recursive
        return [SimpleNamespace(object_name=blob.name) for blob in self._container(bucket_name).list_blobs(name_starts_with=prefix)]

    def get_object(self, bucket_name: str, object_name: str):
        return _BlobResponse(self._container(bucket_name).download_blob(object_name).readall())

    def fput_object(self, bucket_name: str, object_name: str, file_path: str):
        with open(file_path, "rb") as stream:
            self._container(bucket_name).upload_blob(name=object_name, data=stream, overwrite=True)


class _BlobResponse:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass

