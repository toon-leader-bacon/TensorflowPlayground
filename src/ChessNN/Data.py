import subprocess
import wget

DATE = "2016-03"  # The first one that is 1 gb large. More contemporary shards are 30+ gigs
def target_file_name(targeted_date: str) -> str:
    base: str = "lichess_db_standard_rated_"
    file_type: str = ".png.zst"
    return base + targeted_date + file_type


def download_data() -> None:
    # https://database.lichess.org/
    # https://database.lichess.org/standard/lichess_db_standard_rated_2016-03.pgn.zst
    lichess_base_url: str = "https://database.lichess.org/standard/"
    target_date: str = DATE
    full_target_url: str = lichess_base_url + target_file_name(target_date)

    wget.download(full_target_url)


def extract_data(filename: str):
    file_name = target_file_name(DATE)
    decompress_file_cmd: str = f"tar --use-compress-program=unzstd -xvf {file_name}"
    p = subprocess.Popen(decompress_file_cmd.split(), stdout=subprocess.PIPE)
    output, error = p.communicate()
    pass



