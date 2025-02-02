import pandera as pa
from pandera.typing import DataFrame

class DataSchema(pa.DataFrameModel):
    """
    Data schema for the generated data
    """
    product: float = pa.Field()
    weight: float = pa.Field(ge=0)
    volume: float = pa.Field(ge=0)

SimulatedDataSchema = DataFrame[DataSchema]