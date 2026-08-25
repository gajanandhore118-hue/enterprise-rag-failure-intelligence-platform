import re
from pathlib import Path


FIELD_PATTERNS = {
    "document_type": [r"^DOCUMENT TYPE:\s*(.+)$"],
    "document_id": [r"^DOCUMENT ID:\s*(.+)$"],
    "product_family": [r"^PRODUCT FAMILY:\s*(.+)$"],
    "product": [r"^PRODUCT:\s*(.+)$"],
    "component": [r"^COMPONENT:\s*(.+)$"],
    "component_id": [r"^COMPONENT ID:\s*(.+)$"],
    "material": [
        r"^PROPOSED MATERIAL:\s*(.+)$",
        r"^MATERIAL:\s*(.+)$",
        r"^CURRENT MATERIAL:\s*(.+)$",
    ],
    "supplier": [r"^PROPOSED SUPPLIER:\s*(.+)$", r"^SUPPLIER:\s*(.+)$"],
    "revision": [r"^REVISION:\s*(.+)$"],
    "status": [r"^STATUS:\s*(.+)$"],
    "effective_date": [r"^EFFECTIVE DATE:\s*(.+)$", r"^EFFECTIVE:\s*(.+)$"],
}


def normalize_value(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip()


def extract_metadata(text: str, source_file: str) -> dict:
    metadata = {
        "source_file": source_file,
        "document_id": Path(source_file).stem,
        "document_type": "unknown",
        "product_family": None,
        "product": None,
        "component": None,
        "component_id": None,
        "material": None,
        "supplier": None,
        "revision": None,
        "status": None,
        "effective_date": None,
    }

    for field, patterns in FIELD_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
            if match:
                metadata[field] = normalize_value(match.group(1))
                break

    # A simple fallback for the synthetic demo dataset.
    if metadata["product_family"] is None:
        lowered = text.lower()
        if any(x in lowered for x in ["coolant pump", "battery thermal", "cp-420", "cp-500", "cp-600"]):
            metadata["product_family"] = "EV Battery Thermal Management"

    return metadata
