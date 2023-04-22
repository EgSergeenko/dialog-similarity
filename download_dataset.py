import os
import tarfile

import requests

DATASET_URL = (
    'https://dax-cdn.cdn.appdomain.cloud/dax-conversation-similarity/'
    '1.0.0/conversation-similarity.tar.gz'
)


def download_dataset() -> None:
    response = requests.get(DATASET_URL)
    archive_filename = 'tmp.tar.gz'
    with open(archive_filename, 'wb') as archive_file:
        archive_file.write(response.content)

    with tarfile.open(archive_filename) as archive_file:
        members = [
            member for member in archive_file.getmembers()
            if not member.name.startswith('.')
        ]
        archive_file.extractall(members=members)

    os.remove(archive_filename)


if __name__ == '__main__':
    download_dataset()
