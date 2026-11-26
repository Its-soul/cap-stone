"""Consolidated research matrix, comparison, and mentor workbook generator.

Replaces the overlapping legacy generators and writes all outputs strictly to an
ignored output directory (default: ``results/``) instead of the repository root.
"""

import argparse
import csv
import io
from pathlib import Path
import sys
import xml.sax.saxutils as saxutils
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _col_letter(col_idx: int) -> str:
    name = ""
    idx = col_idx
    while True:
        name = chr(65 + (idx % 26)) + name
        idx = (idx // 26) - 1
        if idx < 0:
            break
    return name


def build_xlsx(
    output_path: Path,
    sheet_name: str,
    headers: list[str],
    rows: list[list],
    group_headers: dict[int, str] | None = None,
    col_widths: list[int] | None = None,
) -> None:
    """Write a formatted OpenXML ``.xlsx`` workbook using Python stdlib only."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-officedocument.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        )
        z.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><name val="Calibri"/><sz val="10"/></font>
    <font><b/><name val="Calibri"/><sz val="11"/><color rgb="FFFFFFFF"/></font>
    <font><b/><name val="Calibri"/><sz val="10"/><color rgb="FF1F3A5C"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F3A5C"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFEBF2FA"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/></border>
    <border>
      <left style="thin"><color rgb="FFD3D3D3"/></left>
      <right style="thin"><color rgb="FFD3D3D3"/></right>
      <top style="thin"><color rgb="FFD3D3D3"/></top>
      <bottom style="thin"><color rgb="FFD3D3D3"/></bottom>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="center"/></xf>
  </cellXfs>
</styleSheet>""",
        )
        z.writestr(
            "xl/workbook.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{saxutils.escape(sheet_name)}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        )

        cols_xml = ""
        if col_widths:
            cols_xml = "<cols>" + "".join(
                f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
                for i, w in enumerate(col_widths)
            ) + "</cols>"

        sheet_rows = []
        hdr_cells = [
            f'<c r="{_col_letter(c)}1" t="inlineStr" s="1"><is><t>{saxutils.escape(str(h))}</t></is></c>'
            for c, h in enumerate(headers)
        ]
        sheet_rows.append(f'<row r="1" ht="26" customHeight="1">{"".join(hdr_cells)}</row>')

        row_num = 2
        for idx, row in enumerate(rows):
            if group_headers and idx in group_headers:
                grp_cells = [
                    f'<c r="{_col_letter(c)}{row_num}" t="inlineStr" s="2"><is><t>{saxutils.escape(group_headers[idx]) if c == 0 else ""}</t></is></c>'
                    for c in range(len(headers))
                ]
                sheet_rows.append(f'<row r="{row_num}" ht="22" customHeight="1">{"".join(grp_cells)}</row>')
                row_num += 1
            cells = [
                f'<c r="{_col_letter(c)}{row_num}" t="inlineStr" s="0"><is><t>{saxutils.escape(str(v))}</t></is></c>'
                for c, v in enumerate(row)
            ]
            sheet_rows.append(f'<row r="{row_num}">{"".join(cells)}</row>')
            row_num += 1

        z.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  {cols_xml}
  <sheetData>{"".join(sheet_rows)}</sheetData>
</worksheet>""",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buf.getvalue())
    print(f"[+] Generated XLSX: {output_path} ({output_path.stat().st_size} bytes)")


def write_csv(output_path: Path, headers: list[str], rows: list[list]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[+] Generated CSV: {output_path}")


MATRIX_HEADERS = [
    "Model",
    "Group",
    "Publication Year",
    "Release Year",
    "Architecture",
    "Parameters",
    "Paper URL",
    "Checkpoint URL",
    "Target Task",
    "Dataset",
    "Why Selected",
    "Verified Baseline / Local Result",
    "Laptop Feasibility",
    "NVIDIA Lab Profile",
    "Priority",
]


def _matrix_rows_from_registry():
    from research_benchmark.registry import get_model_registry

    rows = []
    for m in get_model_registry():
        rows.append([
            m.model_name,
            f"{m.group_id}: {m.group_name}",
            m.architecture_publication_year,
            m.checkpoint_release_year,
            m.architecture_paper_title,
            m.parameter_count,
            m.paper_url,
            m.checkpoint_url,
            m.project_objective,
            m.dataset_source,
            m.our_modifications,
            m.historical_results,
            m.laptop_feasibility,
            m.nvidia_feasibility,
            m.priority,
        ])
    return rows


EVOLUTION_HEADERS = [
    "Stage",
    "Model Variant",
    "Backbone",
    "Head Architecture",
    "Dataset & Splits",
    "Training Budget",
    "Threshold (tau)",
    "Observed Metrics",
    "nli_context Confound Status",
    "Execution Status",
]

EVOLUTION_ROWS = [
    [
        "Stage A: Pretrained Backbone",
        "DeBERTa-v3-small (Zero-Shot NLI Proxy)",
        "cross-encoder/nli-deberta-v3-small",
        "Pretrained 3-class NLI head",
        "Pilot provenance test split (16 rows)",
        "0 steps (Pretrained weights)",
        "0.50",
        "Acc: 0.5000, Macro-F1: 0.3333, FN: 8/8",
        "Confounded (no DAG provenance concept)",
        "VERIFIED_EXECUTED",
    ],
    [
        "Stage B: Problem Formulation",
        "DeBERTa-v3-small + Fresh Binary Head",
        "cross-encoder/nli-deberta-v3-small",
        "Seeded 2-layer MLP head (2 classes)",
        "Candidate V2 validation split (64 rows)",
        "0 steps (Untrained prior)",
        "0.50",
        "Random prior prior to SFT",
        "Baseline reference",
        "VERIFIED_BASELINE",
    ],
    [
        "Stage C/D: Candidate V2 Checkpoint",
        "Candidate V2 (Controlled SFT)",
        "cross-encoder/nli-deberta-v3-small",
        "2-layer MLP (768 -> 2)",
        "Controlled suite (256 train / 64 val / 96 final + 384 stress)",
        "256 steps (2 epochs, bs=2, lr=2e-5)",
        "0.20 (Validation-selected)",
        "Val Acc: 87.5% (F1 0.873); Final Acc: 91.7% (F1 0.916, 0 FN); Stress Acc: 82.8% (F1 0.823)",
        "Failed 16/16 nli_context pairs",
        "VERIFIED_FROZEN_CHECKPOINT",
    ],
    [
        "Stage E: Candidate V3 Contrast",
        "Candidate V3 (Root-Explicit Contrast)",
        "cross-encoder/nli-deberta-v3-small",
        "2-layer MLP with explicit closure rules",
        "512 root-explicit contrast rows + 576 links",
        "256 steps configured (not run)",
        "0.20 (Validation selection preserved)",
        "Not executed (max 113/128 tokens, 0 truncation audited)",
        "Designed to address confound via explicit negative closure",
        "PREPARED_NOT_RUN",
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", "matrix", "registry", "evolution"),
        default="all",
        help="Which report artifact(s) to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Destination directory (default: results/)",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.target in ("all", "registry"):
        from research_benchmark.registry import export_registry_csv
        export_registry_csv(str(out / "12_Model_Registry.csv"))

    if args.target in ("all", "matrix"):
        rows = _matrix_rows_from_registry()
        write_csv(out / "12_model_research_matrix.csv", MATRIX_HEADERS, rows)
        build_xlsx(
            out / "12_model_research_matrix.xlsx",
            "12_Model_Research_Matrix",
            MATRIX_HEADERS,
            rows,
        )

    if args.target in ("all", "evolution"):
        write_csv(out / "Custom_V2_V3_Model_Evolution.csv", EVOLUTION_HEADERS, EVOLUTION_ROWS)
        build_xlsx(
            out / "Custom_V2_V3_Model_Evolution.xlsx",
            "V2_V3_Evolution",
            EVOLUTION_HEADERS,
            EVOLUTION_ROWS,
        )


if __name__ == "__main__":
    main()
