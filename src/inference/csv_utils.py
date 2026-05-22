import io
import pandas as pd
from typing import BinaryIO
from app.schemas import ColumnorientedInput


def csv_to_column_oriented(file_stream: BinaryIO) -> ColumnorientedInput:
    df = pd.read_csv(file_stream, encoding="latin1")
    records = df.to_dict(orient="list")
    return ColumnorientedInput(**records)
