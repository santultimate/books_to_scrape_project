from typing import Any

from django.contrib.gis.gdal.base import GDALBase as GDALBase

class Field(GDALBase):
    ptr: Any = ...
    __class__: Any = ...
    def __init__(self, feat: Any, index: Any) -> None: ...
    def as_double(self) -> Any: ...
    def as_int(self, is_64: bool = ...) -> Any: ...
    def as_string(self) -> Any: ...
    def as_datetime(self) -> Any: ...
    @property
    def is_set(self) -> Any: ...
    @property
    def name(self) -> Any: ...
    @property
    def precision(self) -> Any: ...
    @property
    def type(self) -> Any: ...
    @property
    def type_name(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    @property
    def width(self) -> Any: ...

class OFTInteger(Field):
    @property
    def value(self) -> Any: ...
    @property
    def type(self) -> Any: ...

class OFTReal(Field):
    @property
    def value(self) -> Any: ...

class OFTString(Field): ...
class OFTWideString(Field): ...
class OFTBinary(Field): ...

class OFTDate(Field):
    @property
    def value(self) -> Any: ...

class OFTDateTime(Field):
    @property
    def value(self) -> Any: ...

class OFTTime(Field):
    @property
    def value(self) -> Any: ...

class OFTInteger64(OFTInteger): ...
class OFTIntegerList(Field): ...
class OFTRealList(Field): ...
class OFTStringList(Field): ...
class OFTWideStringList(Field): ...
class OFTInteger64List(Field): ...

OGRFieldTypes: Any
ROGRFieldTypes: Any