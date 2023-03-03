import wget
import zstandard as zstd
import subprocess


def target_file_name(targeted_date: str) -> str:
    base: str = "lichess_db_standard_rated_"
    file_type: str = ".png.zst"
    return base + targeted_date + file_type


def download_data() -> None:
    # https://database.lichess.org/
    # https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst
    lichess_base_url: str = "https://database.lichess.org/standard/"
    target_date: str = "2023-01"  # TODO: Make this a variable/ dynamic?
    full_target_url: str = lichess_base_url + target_file_name(target_date)

    wget.download(full_target_url)


def extract_data(filename: str):
    # TODO: Extract the data, either into a file or directly into an object of some kind
    file_name = target_file_name("2023-1")
    decompress_file_cmd: str = f"tar --use-compress-program=unzstd -xvf {file_name}"
    p = subprocess.Popen(decompress_file_cmd.split(), stdout=subprocess.PIPE)
    output, error = p.communicate()
    pass
