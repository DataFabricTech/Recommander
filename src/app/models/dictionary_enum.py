from enum import Enum


class DictionaryKeys(Enum):
    NAME = "name"
    DESCRIPTION = 'description'
    DISPLAY_NAME = 'displayName'
    TAGS = 'TAGS'
    ID = 'id'

    SERVICE = 'service'

    DATA_TYPE = "dataType"
    COLUMNS = "columns"
    VALUES = "values"

    TAG_FQN = 'tagFQN'
    TABLE_CONSTRAINTS = 'tableConstraints'
    CONSTRAINT_TYPE = 'constraintType'
    REFERRED_COLUMNS = 'referredColumns'
    FULLY_QUALIFIED_NAME = 'fullyQualifiedName'
    SAMPLE_DATA = 'sampleData'
    ROWS = 'rows'

    EMBEDDINGS_RESULT = "embeddings_result"
