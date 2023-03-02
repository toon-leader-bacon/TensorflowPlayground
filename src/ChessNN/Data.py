import wget
import zstandard as zstd


def download_data():
    # https://database.lichess.org/
    # https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst
    lichess_base_url: str = "https://database.lichess.org/standard/lichess_db_standard_rated_"
    target_date: str = "2023-01"  # TODO: Make this a variable/ dynamic?
    lichess_file_type: str = ".png.zst"

    full_target_url: str = lichess_base_url + target_date + lichess_file_type
    wget.download(full_target_url)


def extract_data(filename: str):
    # TODO: Extract the data, either into a file or directly into an object of some kind
    pass
