
from pathlib import Path
import os
import re


from globalVars import folder1, folder2


def rename_correct_wrong_raw_datafiles():
    words = ['ακαδημαϊκά', 'αλάτια', 'αρδευτικός', 'αυθαιρεσίες', 'βοοειδές', 'δεσμεύω',
                'δραστηριότητες', 'εγκλείει', 'εκπαίδευση', 'εκσυγχρονισμένος', 'επιχειρήματα',
                'ηλεκτρική', 'λουλούδια', 'μετεγγραφές', 'παυσίπονο', 'περιβάλλον', 'ποίημα',
                'πρόσκληση', 'σκουπίδια', 'σταφιδιασμένα', 'συμπληρώσουμε', 'ταξιδεύουμε',
                'τήβεννος', 'υπεύθυνος']


    folder_path = Path(folder1+ folder2)

    folders = [
        folder
        for folder in folder_path.iterdir()
        if folder.is_dir() and folder.name in words
    ]

    for folder in folders:
        correct_folder = [
            path
            for path in folder.iterdir()
            if path.is_dir() and path.name.endswith("_correct")
        ][0]

        wrong_folder = [
            path
            for path in folder.iterdir()
            if path.is_dir() and path.name.endswith("_wrong")
        ][0]
        
        correct_files = [
            path
            for path in correct_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        ]
        
        wrong_files = [
            path
            for path in wrong_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        ]

        new_files = [
            file.rename(file.with_name("1" + file.name))
            for file in correct_files + wrong_files
        ]


def windows_path(path):
    """Allow Windows to work with paths longer than 260 characters."""
    path = str(Path(path).resolve())

    if os.name == "nt" and not path.startswith("\\\\?\\"):
        if path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path[2:]

        return "\\\\?\\" + path

    return path


def safe_rename(old_path, new_path):
    old_path = windows_path(old_path)
    new_path = windows_path(new_path)

    if not os.path.exists(old_path):
        # It may already have been renamed during a previous partial run
        if os.path.exists(new_path):
            return True

        print(f"Missing, skipped: {old_path}")
        return False

    if os.path.exists(new_path):
        print(f"Destination already exists, skipped: {new_path}")
        return False

    os.rename(old_path, new_path)
    return True


from pathlib import Path
import os
import re


def windows_path(path):
    """Allow Windows to work with paths longer than 260 characters."""
    path = str(Path(path).resolve())

    if os.name == "nt" and not path.startswith("\\\\?\\"):
        if path.startswith("\\\\"):
            return "\\\\?\\UNC\\" + path[2:]

        return "\\\\?\\" + path

    return path


def safe_rename(old_path, new_path):
    old_path = windows_path(old_path)
    new_path = windows_path(new_path)

    if not os.path.exists(old_path):
        # It may already have been renamed during a previous partial run
        if os.path.exists(new_path):
            return True

        print(f"Missing, skipped: {old_path}")
        return False

    if os.path.exists(new_path):
        print(f"Destination already exists, skipped: {new_path}")
        return False

    os.rename(old_path, new_path)
    return True


def rename_individual_folders():
    base_folder = Path(folder1) / "individual_decodewordspseudo_extra"

    for subject in base_folder.glob("subject_*"):
        if not subject.is_dir():
            continue

        marker = subject / ".renamed"

        if marker.exists():
            continue

        old_id = subject.name.split("_")[-1]

        if not old_id.isdigit():
            continue

        new_id = "1" + old_id

        # Replace only the exact ID.
        # It will not match 13824 inside 113824.
        id_pattern = re.compile(
            rf"(?<!\d){re.escape(old_id)}(?!\d)"
        )

        # First rename all files
        for root, folders, files in os.walk(subject):
            root = Path(root)

            for filename in files:
                new_filename = id_pattern.sub(new_id, filename)

                if new_filename == filename:
                    continue

                old_path = root / filename
                new_path = root / new_filename

                if safe_rename(old_path, new_path):
                    print(f"File: {filename} -> {new_filename}")

        # Then rename folders from deepest to highest
        for root, folders, files in os.walk(subject, topdown=False):
            root = Path(root)

            for foldername in folders:
                new_foldername = id_pattern.sub(new_id, foldername)

                if new_foldername == foldername:
                    continue

                old_path = root / foldername
                new_path = root / new_foldername

                if safe_rename(old_path, new_path):
                    print(f"Folder: {foldername} -> {new_foldername}")

        marker.touch()

        # Rename subject folder last
        new_subject = subject.with_name(f"subject_{new_id}")

        if safe_rename(subject, new_subject):
            print(f"Subject: {subject.name} -> {new_subject.name}")


rename_individual_folders()


rename_individual_folders()