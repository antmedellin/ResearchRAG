"""Small provenance helpers for canonical fields."""
from datetime import datetime, timezone
def set_field(record, field, value, source, source_record=None, overwrite=False):
    if value in (None,"",[],{}): return False
    if record.get(field) not in (None,"",[],{}) and not overwrite: return False
    record[field]=value
    record.setdefault("provenance",{})[field]={"source":source,"source_record":source_record,
      "observed_at":datetime.now(timezone.utc).isoformat()}
    return True
def union_field(record, field, values, source):
    merged=list(dict.fromkeys([*record.get(field,[]),*values])); record[field]=merged
    record.setdefault("provenance",{})[field]={"source":source,"mode":"union"}; return merged
