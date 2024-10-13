import os

import requests


def download_file(url: str, save_path: str = None, save_folder: str = '', file_name: str = None):
    if save_path is None:
        if file_name is None:
            file_name = os.path.basename(url)
        max_length_filename = os.pathconf('/', 'PC_NAME_MAX')
        if len(file_name) > max_length_filename:
            split_text = os.path.splitext(file_name)
            file_name = split_text[0][: max_length_filename - len(split_text[1])] + split_text[1]
        os.makedirs(save_folder, exist_ok=True)
        save_path = f'{save_folder}/{file_name}'

    response = requests.get(url, timeout=20)
    assert response.status_code == 200, f"Failed to reach {url} with the status code of {response.status_code}"
    with open(save_path, 'wb') as file:
        file.write(response.content)

    # Download success
    assert os.path.isfile(save_path), f"Downloaded file is not found in: {save_path}"
    return save_path
