import argparse
import fnmatch
import os
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient


def client() -> BlobServiceClient:
    connection_string = os.getenv("AZURITE_CONNECTION_STRING")
    if connection_string and not connection_string.startswith("replace-with"):
        return BlobServiceClient.from_connection_string(connection_string)
    return BlobServiceClient(
        account_url=os.getenv("AZURITE_BLOB_ENDPOINT", "http://127.0.0.1:10000/devstoreaccount1"),
        credential={
            "account_name": os.getenv("AZURITE_ACCOUNT_NAME", "devstoreaccount1"),
            "account_key": os.getenv("AZURITE_ACCOUNT_KEY", ""),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a directory to an Azurite Blob container")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional blob-name prefix (for example, 'images' or 'labels')",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only upload relative paths matching this glob; may be repeated",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Skip relative paths matching this glob; may be repeated",
    )
    parser.add_argument(
        "--reset-container",
        action="store_true",
        help="Delete and recreate the target container before uploading",
    )
    parser.add_argument(
        "--require-empty-prefix",
        metavar="PREFIX",
        help="Refuse to upload if any blob already exists below this prefix",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"source is not a directory: {args.source}")

    prefix = args.prefix.strip("/")
    if prefix:
        prefix_path = Path(prefix)
        if prefix_path.is_absolute() or ".." in prefix_path.parts:
            parser.error("prefix must be a relative blob path without '..'")

    service = client()
    container = service.get_container_client(args.container)
    if args.reset_container:
        try:
            container.delete_container()
            print(f"Deleted existing container: {args.container}")
        except ResourceNotFoundError:
            pass
        container = service.get_container_client(args.container)

    try:
        container.create_container()
    except Exception as exc:
        if "ContainerAlreadyExists" not in str(exc):
            raise

    if args.require_empty_prefix:
        guard_prefix = args.require_empty_prefix.strip("/")
        existing = next(container.list_blobs(name_starts_with=guard_prefix), None)
        if existing is not None:
            parser.error(
                f"refusing to overwrite immutable prefix {guard_prefix!r}; "
                f"existing blob: {existing.name}"
            )
    for path in sorted(args.source.rglob("*")):
        if path.is_file():
            relative_name = path.relative_to(args.source).as_posix()
            if args.include and not any(
                fnmatch.fnmatch(relative_name.lower(), pattern.lower())
                for pattern in args.include
            ):
                continue
            if any(
                fnmatch.fnmatch(relative_name.lower(), pattern.lower())
                for pattern in args.exclude
            ):
                continue
            blob_name = f"{prefix}/{relative_name}" if prefix else relative_name
            with path.open("rb") as stream:
                container.upload_blob(name=blob_name, data=stream, overwrite=True)
            print(f"Uploaded {path} -> {blob_name}")


if __name__ == "__main__":
    main()
