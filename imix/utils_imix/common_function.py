from typing import Dict


def update_d1_with_d2(d1: Dict, d2: Dict):  # update d1 with d2
    for k, v in d2.items():
        d1[k] = d1.get(k, v)
