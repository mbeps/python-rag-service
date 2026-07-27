from typing import BinaryIO
from minio import Minio
from minio.error import S3Error


class MinIOManager:
    """
    Manager for MinIO object storage operations.
    """

    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = False
    ) -> None:
        """
        Initializes the MinIO client.

        Args:
            endpoint: MinIO server endpoint (host:port).
            access_key: Access key for authentication.
            secret_key: Secret key for authentication.
            secure: Whether to use HTTPS.

        Returns:
            None
        """
        self.client: Minio = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    async def ensure_bucket(self, bucket_name: str) -> None:
        """
        Ensures a bucket exists, creating it if it doesn't.

        Args:
            bucket_name: The name of the bucket to ensure.

        Returns:
            None
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except S3Error as e:
            # Re-raise or handle specific S3 errors if needed
            raise e

    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Uploads a file to a specific bucket.

        Args:
            bucket_name: The name of the bucket.
            object_name: The name of the object in the bucket.
            data: Binary IO stream of the file content.
            length: Length of the data in bytes.
            content_type: Content type of the file.

        Returns:
            str: The object name (path) if successful.

        Raises:
            ValueError: If the file size is invalid.
            S3Error: For MinIO specific errors.
        """
        if length <= 0:
            raise ValueError("File size must be greater than zero")

        try:
            self.client.put_object(
                bucket_name, object_name, data, length, content_type=content_type
            )
            return object_name
        except S3Error as e:
            raise e

    async def get_presigned_url(
        self, bucket_name: str, object_name: str, expires_in: int = 3600
    ) -> str:
        """
        Generates a presigned URL for a specific object.

        Args:
            bucket_name: The name of the bucket.
            object_name: The name of the object.
            expires_in: Time in seconds until the URL expires.

        Returns:
            The presigned URL string.
        """
        from datetime import timedelta

        try:
            url = self.client.get_presigned_url(
                "GET", bucket_name, object_name, expires=timedelta(seconds=expires_in)
            )
            return url
        except S3Error as e:
            raise e
