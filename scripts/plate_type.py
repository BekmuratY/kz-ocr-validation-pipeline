from __future__ import annotations

import re


def detect_plate_type(plate_string: str) -> str:
    s = (plate_string or "").replace(" ", "").replace("-", "").upper()

    allowed = set("ABCEHKMOPTXY0123456789")
    if any(ch not in allowed for ch in s):
        return "INVALID"

    # NEW_STANDARD: call existing checker as-is if available.
    try:
        from scripts.postprocess import detect_plate_format

        if detect_plate_format(s) == "std_private":
            return "NEW_STANDARD"
    except Exception:
        pass

    if re.fullmatch(r"^[ABCEHKMOPTXY]{1}[0-9]{3}[ABCEHKMOPTXY]{2}$", s):
        return "OLD_STANDARD"

    if re.fullmatch(r"^CD[0-9]{5}$", s) or re.fullmatch(r"^D[0-9]{5}$", s):
        return "DIPLOMATIC"

    if re.fullmatch(r"^[0-9]{3}[ABCEHKMOPTXY]{3}$", s):
        return "TRANSIT"

    if re.fullmatch(r"^[ABCEHKMOPTXY]{2}[0-9]{4}$", s) or re.fullmatch(
        r"^[0-9]{4}[ABCEHKMOPTXY]{2}$", s
    ):
        return "TRAILER"

    return "INVALID"
