import argparse


def parse() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                        help="읽을 데이터셋 파일 이름")

    args = parser.parse_args()
    return args.file
