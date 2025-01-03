from pathlib import Path

def get_unique_filename(file_path):
    """
    겹치지 않는 파일 이름을 반환합니다. (Pathlib 버전)
    """
    file = Path(file_path)
    counter = 1
    new_file = file

    while new_file.exists():
        new_file = file.with_name(f"{file.stem}({counter}){file.suffix}")
        counter += 1

    return str(new_file)