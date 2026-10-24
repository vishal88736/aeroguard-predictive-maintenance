"""Small synthetic parser fixtures only; never written to the project dataset."""

import io
import zipfile

from fleetpm.config import SUBSETS


def row(unit=1,cycle=1,sensor=500):
    return " ".join(map(str,[unit,cycle,0,0,100]+[sensor]*21))+"\n"


def archive_payload(engines=2,cycles=3,prefix="CMaps/",sensor=500):
    content=io.BytesIO()
    with zipfile.ZipFile(content,"w",zipfile.ZIP_DEFLATED) as archive:
        for subset in SUBSETS:
            records="".join(row(unit,cycle,sensor) for unit in range(1,engines+1) for cycle in range(1,cycles+1))
            archive.writestr(prefix+f"train_{subset}.txt",records)
            archive.writestr(prefix+f"test_{subset}.txt",records)
            archive.writestr(prefix+f"RUL_{subset}.txt","15\n"*engines)
    return content.getvalue()
