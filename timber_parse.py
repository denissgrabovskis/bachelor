from __future__ import annotations

import re
import math
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from openpyxl.cell import cell, Cell

from openpyxl.worksheet.worksheet import Worksheet
from openpyxl import load_workbook

from dataclasses import dataclass
from typing import Optional


# ----------------------------
# SQLite schema (hardcoded)
# ----------------------------


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS materials (
  id TEXT PRIMARY KEY,
  brand       TEXT,
  category    TEXT,
  quality     TEXT,
  height_m    REAL,
  width_m     REAL,
  length_m    REAL,
  row         REAL
);

CREATE TABLE IF NOT EXISTS monthly (
  material_id TEXT NOT NULL,
  period       TEXT NOT NULL, -- yyyy-mm
  
  price_purchase    REAL,
  price_m3          REAL,
  price_pcs         REAL,
  price_m2          REAL,
  
  qty_pcs   REAL,
  qty_m3    REAL,
  qty_m2    REAL,
  
  sold_pcs      REAL,
  sold_m3       REAL,
  sold_eur      REAL,
  
  received_pcs  REAL,
  received_m3   REAL,
  
  source_file  TEXT,
  sheet_name   TEXT,

  PRIMARY KEY (material_id, period)
);

CREATE TABLE IF NOT EXISTS sales (
  material_id   TEXT NOT NULL,
  period        TEXT NOT NULL,

  sale_pcs  REAL,
  sale_eur  REAL,

  PRIMARY KEY (material_id, period)
);
"""


# ----------------------------
# Small helpers
# ----------------------------

formulas_workbook = None
def load_workbook_with_formulas(source):
    global formulas_workbook
    formulas_workbook = load_workbook(source, data_only=False)
    return load_workbook(source, data_only=True)
def get_formula(cell: Cell):
    global formulas_workbook
    return str(formulas_workbook[cell.parent.title][cell.coordinate].value)

def normalize_text(value: Any) -> str:
    """Uppercase + strip + remove diacritics + compress whitespace."""
    if value is None:
        return ""
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_latvian_number(value: Any) -> Optional[float]:
    """
    Parses:
      - 0,027
      - 1 614,94
      - 1614.94
      - numeric types
    Returns None for empty/invalid.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    s = s.replace(" ", "")  # thousands separators
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        s = s.replace(",", "")  # rare, but safe

    try:
        return float(s)
    except ValueError:
        return None


_ALLOWED_EXPR = re.compile(r"^[0-9\.\+\-\*\/\(\) ,]+$")


def parse_number_or_expression(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """
    For "100+50" returns (150.0, "100+50")
    For "36" returns (36.0, None)
    For empty returns (None, None)
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value), None

    raw = str(value).strip()
    if raw == "":
        return None, None

    expr = raw.replace(" ", "").replace(",", ".")
    if _ALLOWED_EXPR.match(expr) and any(op in expr for op in "+-*/()"):
        try:
            return float(eval(expr, {"__builtins__": {}}, {})), raw
        except Exception:
            return None, raw

    return parse_latvian_number(raw), None


def parse_period_from_title(sheet_title: str) -> Optional[str]:
    m = re.fullmatch(r"((0[1-9])|1[0-2])\.((19|20)[0-9]{2})", sheet_title.strip(' .')) # parses dd.yyyy 01.1900-12.2099
    if not m:
        return None
    month = int(m.group(1))
    year = int(m.group(3))
    return f"{year:04d}-{month:02d}"

get_headers_template = lambda: {
    # material table
    'KVALITĀTE': 'quality',
    'AUGSTUMS (m)': 'height_m', 'BIEZUMS (m)': 'height_m',
    'PLATUMS (m)': 'width_m',
    'GARUMS (m)': 'length_m',


    # monthly data
    'IEPIRKUMA CENA': 'price_purchase',
    'Cena m3': 'price_m3',
    'Cena 1 gb': 'price_pcs',
    'Cena m2': 'price_m2',

    'DAUDZUMS (gb.)': 'qty_pcs',
    'Kopā m3': 'qty_m3',
    'Kopa m2': 'qty_m2',

    'PĀRDOTS (gb.)': 'sold_pcs',
    'PĀRDOTS (€)': 'sold_eur',

    'SAŅEMAM (gb.)': 'received_pcs',
    'SAŅEMAM (m3)': 'received_m3',

    # ignored
    'm3 (1 gb)': 'm3_pcs',
    'm2(1gb)': 'm2_pcs',
    'ATLIKUMS (gb.)': 'leftovers_pcs', # duplicate of previous month
    'ATLIKUMS (m3)': 'leftovers_m3', # duplicate of previous month
    'ATLIKUMS (m2)': 'leftovers_m2', # duplicate of previous month
    'ATLIKUMS (€)': 'leftovers_eur', # duplicate of previous month
    'PĀRDOTS (m3)': 'sold_m3', # redundant, difficult to parse
}

headers_template = None
def remember_header(worksheet: Worksheet) -> Optional[int]:
    global headers_template
    headers_template = get_headers_template()
    for row_index in range(1, 10):
        headers = [normalize_text(cell.value) for cell in worksheet[row_index]]
        if ('KVALITĀTE' not in headers): continue

        if 'AUGSTUMS (m)' in headers:
            headers_template.pop('BIEZUMS (m)')
        if 'BIEZUMS (m)' in headers:
            headers_template.pop('AUGSTUMS (m)')

        headers_template = {
            field: headers.index(header) if header in headers else None
            for header, field in headers_template.items()
        }
        return row_index
    return None

def warn(message):
    print(message) # TODO

def build_brand_for_row_map(worksheet: Worksheet) -> Dict[int, str]:
    brand_at_row: Dict[int, str] = {}

    for merged_range in worksheet.merged_cells.ranges:
        if merged_range.min_col != 1 or merged_range.max_col != 1:
            continue

        brand_value = normalize_text(worksheet.cell(merged_range.min_row, 1).value)

        for row_index in range(merged_range.min_row, merged_range.max_row + 1):
            brand_at_row[row_index] = brand_value

    return brand_at_row

def tryFloat(value):
    try:
        return float(value.replace(',', '.').replace(' ', ''))
    except:
        return value

def parse_record(row, brand, category):
    values = [tryFloat(normalize_text(cell.value)) for cell in row]
    formulas = [get_formula(cell) for cell in row]

    record = dict()
    for field, header_col in headers_template.items():
        if header_col is not None:
            record[field] = values[header_col]
            record[field+'_formula'] = formulas[header_col]
        else:
            record[field] = None
            record[field+'_formula'] = None
    record = type('',(object,),record)() # https://stackoverflow.com/a/29480317

    if not record.height_m or not record.width_m or not record.length_m: return None
    record.id = f"{brand}|{category}|{record.quality or 'empty'}|{record.height_m:.3f}|{record.width_m:.3f}|{record.length_m:.3f}"
    record.sold_eur_formula = record.sold_eur_formula if '*' not in record.sold_eur_formula else f'={record.sold_eur}'

    return record


def round_dim(x: float) -> float:
    return float(f"{x:.3f}")


def make_id(
    brand: str,
    category: str,
    quality: Optional[str],
    height_m: float,
    width_m: float,
    length_m: float,
) -> str:
    # identity includes brand+category as you requested
    q = normalize_text(quality)
    b = normalize_text(brand)
    c = normalize_text(category)
    h, w, l = round_dim(height_m), round_dim(width_m), round_dim(length_m)
    return f"{b}|{c}|{q}|{h:.3f}|{w:.3f}|{l:.3f}"


def is_totals_row(text_value: str) -> bool:
    return normalize_text(text_value).startswith("KOPA")  # catches KOPĀ / KOPA / KOPĀ VISS / ...


def split_plus_list(raw: str) -> List[str]:
    """
    "100+50+ 25" -> ["100", "50", "25"]
    """
    raw = raw.replace(" ", "")
    return [t for t in raw.split("+") if t != ""]


# ----------------------------
# Core parse
# ----------------------------

def parse_worksheet(
    worksheet: Worksheet,
    period: str,
    workbook_name: str,
    header_row_index: int,
) -> Tuple[
    List[Dict[str, Any]],  # materials upserts
    List[Dict[str, Any]],  # monthly upserts
    List[Dict[str, Any]],  # sales_lines upserts
]:
    # Read headers from the header row, normalized
    brands_by_row = build_brand_for_row_map(worksheet)

    materials = dict()
    materials_rows: List[Dict[str, Any]] = []
    monthly_rows: List[Dict[str, Any]] = []
    sales_rows: List[Dict[str, Any]] = []

    current_row = header_row_index
    while True:
        current_row+=1

        if current_row not in brands_by_row:
            break
        else:
            current_brand = brands_by_row[current_row]

        current_category = normalize_text(worksheet[current_row][1].value)

        while True:
            current_row += 1
            if (worksheet[current_row][1].value and 'KOPĀ' in worksheet[current_row][1].value): break

            record = parse_record(worksheet[current_row], current_brand, current_category)
            if (record is None): continue

            if record.id in materials:
                raise ValueError(f"Duplicate material in one sheet: {record.id} (row {current_row}, sheet {worksheet.title})")

            # ---- materials upsert ----
            materials_rows.append({
                "id": record.id,
                "brand": current_brand,
                "category": current_category,
                "quality": record.quality,
                "height_m": record.height_m,
                "width_m": record.width_m,
                "length_m": record.length_m,
                "row": current_row,
            })

            monthly_rows.append({
                "material_id": record.id,
                "period": period,

                "price_purchase": record.price_purchase,
                "price_m3": record.price_m3,
                "price_pcs": record.price_pcs,
                "price_m2": record.price_m2,

                "qty_pcs": record.qty_pcs,
                "qty_m3": record.qty_m3,
                "qty_m2": record.qty_m2,

                "sold_pcs": record.sold_pcs,
                "sold_m3": record.sold_m3,
                "sold_eur": record.sold_eur,

                "received_pcs": record.received_pcs,
                "received_m3": record.received_m3,

                "source_file": workbook_name,
                "sheet_name": worksheet.title,
            })

            if not record.sold_pcs: continue


            sale_psc = [tryFloat(v) for v in record.sold_pcs_formula[1:].split('+')] if record.sold_pcs_formula else record.sold_pcs
            sale_eur = [tryFloat(v) for v in record.sold_eur_formula[1:].split('+')] if record.sold_eur_formula else record.sold_eur
            if (len(sale_psc) != len(sale_eur)):
                raise ValueError(f'PĀRDOTS (gb.) and PĀRDOTS (€) do not match for record {record.id} from {period}')
            for pcs, eur in zip(sale_psc, sale_eur):
                sales_rows.append({
                    "material_id": record.id,
                    "period": period,

                    "sale_pcs": pcs,
                    "sale_eur": eur,
                })

    return materials_rows, monthly_rows, sales_rows


# ----------------------------
# Upserts
# ----------------------------

def upsert_many(connection: sqlite3.Connection, table: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    column_list = ", ".join(columns)

    sql = f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES ({placeholders})"
    connection.executemany(sql, ([row.get(c) for c in columns] for row in rows))


def delete_sales_lines_for(connection: sqlite3.Connection, keys: List[Tuple[str, str]]) -> None:
    """
    If a (id, period) already exists, we must delete old sales_lines,
    otherwise leftover old line_no rows remain.
    """
    if not keys:
        return
    connection.executemany(
        "DELETE FROM sales_lines WHERE id = ? AND period = ?",
        keys
    )



sources = list(Path("sources").glob("*.xlsx"))
with sqlite3.connect("timber.sqlite") as db:
    db.executescript(SCHEMA_SQL)
    for source in sources:
        workbook = load_workbook_with_formulas(source)

        for worksheet in workbook.worksheets:
            period = parse_period_from_title(worksheet.title)
            if period is None:
                continue  # per your plan: skip non mm.yyyy sheets

            header_row_index = remember_header(worksheet)
            if header_row_index is None:
                warn(f'Worksheet {period} does not have header row')
                continue

            materials_rows, monthly_rows, sales_rows = parse_worksheet(
                worksheet=worksheet,
                period=period,
                workbook_name=source.name,
                header_row_index=header_row_index,
            )

            # Upserts
            upsert_many(db, "materials", materials_rows)
            upsert_many(db, "monthly", monthly_rows)
            upsert_many(db, "sales", sales_rows)

# TODO: ignore empty lines (see 07.2025)
# TODO: ignore records about written off sales
# TODO: