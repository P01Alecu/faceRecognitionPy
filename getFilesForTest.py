import os

def list_files_in_folder(folder_path):
    """
    Listează toate fișierele și folderele din folderul dat.

    Args:
    folder_path: Calea către folderul care trebuie listat.

    Returns:
    O lista [nume_imagine, nume_fisier_in_care_se_afla_imaginea]
    """
    files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isdir(full_path):
            tempList = list_files_in_folder(full_path)
            files.extend(tempList)
        else:
            files.append((file, os.path.dirname(full_path)))

    return files

def get_substring_before_last_slash(string):
    """
    Returnează subșirul dintre ultimele două '/' din string.

    Args:
    string: Stringul din care se returnează subșirul.

    Returns:
    Subșirul dintre ultimele două '/' din string.
    """
    last_slash_index = string.rfind("/")
    substrings = string.rsplit("/", last_slash_index + 1)
    return substrings[-1]
