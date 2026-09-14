"""The DIG-TAG collaboration interface: DT = (D, T, Z, U), built on the
DIG and the TAG, which stay separate records.

  digtag.py  DIGTAG, the DIG interface with the TAG's tools in its write path: a task call
             transforms T, and what it hands back is the event D records
"""

from .digtag import SCHEMA, TAG_ORIGIN, DIGTAG

__all__ = ["DIGTAG", "SCHEMA", "TAG_ORIGIN"]
