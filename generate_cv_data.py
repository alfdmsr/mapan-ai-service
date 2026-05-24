"""
MAPAN synthetic CV dataset generator (Gemini labeling only).

Pilot: 1 CV per sector (indices 1).
Full scale: 600 CVs per sector x 5 sectors = 3000 total.

Usage:
  python generate_cv_data.py --mode pilot
  python generate_cv_data.py --mode full
  python generate_cv_data.py --mode full --sector Teknologi
  python generate_cv_data.py --mode full --start 10 --end 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "synthetic"
FINGERPRINT_DIR = OUTPUT_DIR / ".fingerprints"
PROGRESS_PATH = OUTPUT_DIR / "generation_progress.json"
FAILED_LOG_PATH = OUTPUT_DIR / "failed_generations.log"

PILOT_SECTORS = [
    "Teknologi",
    "Keuangan_Admin",
    "Manufaktur_Engineering",
    "Kreatif_Media",
    "Sales_Pelayanan",
]

SECTOR_ID_PREFIX: dict[str, str] = {
    "Teknologi": "cv_tek",
    "Keuangan_Admin": "cv_keu",
    "Manufaktur_Engineering": "cv_man",
    "Kreatif_Media": "cv_kre",
    "Sales_Pelayanan": "cv_sal",
}

CVS_PER_SECTOR = 600
REQUEST_DELAY_SECONDS = 3
GENERATION_TEMPERATURE = 0.42

ALLOWED_LABELS = {"O", "B-SKILL", "I-SKILL", "B-ROLE", "I-ROLE"}
MAX_RETRIES = 5
MAX_LABEL_PAD = 5
MAX_DUPLICATE_RETRIES = 4
MIN_B_SKILL = 4
MIN_B_ROLE = 1
MIN_TOKENS = 55

INDONESIAN_CITIES = [
    "Jakarta",
    "Bandung",
    "Surabaya",
    "Yogyakarta",
    "Semarang",
    "Medan",
    "Makassar",
    "Denpasar",
    "Malang",
    "Bekasi",
    "Tangerang",
    "Palembang",
]

COMPANY_CONTEXTS = [
    "startup teknologi di BSD",
    "perusahaan manufaktur di kawasan industri Cikarang",
    "BUMN di Jakarta",
    "bank swasta nasional",
    "UMKM retail berkembang",
    "perusahaan FMCG multinasional",
    "lembaga pemerintah daerah",
    "konsultan profesional",
    "perusahaan logistik e-commerce",
    "studio kreatif digital",
]

EXPERIENCE_LEVELS = [
    ("fresh graduate", "0-1 tahun", "lulusan baru mencari pekerjaan pertama"),
    ("junior", "1-3 tahun", "sudah punya pengalaman kerja terbatas"),
    ("mid-level", "3-6 tahun", "mandiri menangani tugas inti"),
    ("senior", "6-10 tahun", "memimpin proyek atau mentoring junior"),
    ("lead", "10+ tahun", "pemimpin tim atau spesialis senior"),
]

SECTOR_REALISM: dict[str, dict[str, object]] = {
    "Teknologi": {
        "roles": [
            "Software Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Data Analyst",
            "DevOps Engineer",
            "QA Engineer",
            "Product Manager",
            "IT Support",
            "Mobile Developer",
            "Machine Learning Engineer",
        ],
        "skills": [
            "Python",
            "Java",
            "JavaScript",
            "React",
            "Node.js",
            "SQL",
            "PostgreSQL",
            "Docker",
            "Kubernetes",
            "AWS",
            "Git",
            "REST API",
            "TensorFlow",
            "Figma",
            "Scrum",
        ],
        "context": (
            "Konteks Indonesia: startup Unicorn, bank digital, outsourcing IT, "
            "perusahaan e-commerce, bootcamp graduate, remote hybrid."
        ),
    },
    "Keuangan_Admin": {
        "roles": [
            "Staf Administrasi Keuangan",
            "Akuntan",
            "Finance Staff",
            "Staff HR",
            "Office Administrator",
            "Tax Officer",
            "Payroll Staff",
            "Auditor Junior",
            "Collection Officer",
            "Receptionist",
        ],
        "skills": [
            "Microsoft Excel",
            "SAP",
            "MYOB",
            "Accurate",
            "pembukuan",
            "rekonsiliasi bank",
            "pajak PPh",
            "BPJS",
            "invoice",
            "arsip dokumen",
            "komunikasi kantor",
            "Google Sheets",
        ],
        "context": (
            "Konteks Indonesia: kantor cabang bank, KAP, perusahaan ekspor-impor, "
            "UMKM, apotek chain, properti, administrasi ruko."
        ),
    },
    "Manufaktur_Engineering": {
        "roles": [
            "Insinyur Produksi",
            "Quality Control Inspector",
            "Maintenance Technician",
            "Process Engineer",
            "Production Supervisor",
            "Teknisi Mesin",
            "Plant Operator",
            "Safety Officer",
            "Industrial Engineer",
            "Warehouse Supervisor",
        ],
        "skills": [
            "Lean Manufacturing",
            "Six Sigma",
            "AutoCAD",
            "PLC",
            "SAP PP",
            "GMP",
            "kalibrasi alat ukur",
            "preventive maintenance",
            "SOP produksi",
            "5S",
            "ISO 9001",
            "analisis root cause",
        ],
        "context": (
            "Konteks Indonesia: pabrik makanan, otomotif, tekstil, elektronik, "
            "kawasan industri Jababeka, Cikarang, Surabaya, shift kerja."
        ),
    },
    "Kreatif_Media": {
        "roles": [
            "Content Creator",
            "Social Media Specialist",
            "Graphic Designer",
            "Video Editor",
            "Copywriter",
            "Digital Marketing Specialist",
            "Photographer",
            "UI Designer",
            "Motion Graphic Artist",
            "Public Relations Staff",
        ],
        "skills": [
            "Adobe Photoshop",
            "Adobe Premiere Pro",
            "Canva",
            "copywriting",
            "SEO",
            "Google Analytics",
            "Instagram Ads",
            "TikTok content",
            "storytelling",
            "branding",
            "After Effects",
            "content strategy",
        ],
        "context": (
            "Konteks Indonesia: agency digital, influencer management, brand lokal, "
            "kampanye Ramadan, marketplace live commerce, media sosial."
        ),
    },
    "Sales_Pelayanan": {
        "roles": [
            "Sales Executive",
            "Customer Service Specialist",
            "Telesales",
            "Retail Supervisor",
            "Field Sales",
            "Relationship Officer",
            "Kasir Supervisor",
            "Business Development",
            "Promotor",
            "Guest Relation Officer",
        ],
        "skills": [
            "negosiasi",
            "CRM",
            "penjualan B2B",
            "penjualan B2C",
            "handling complaint",
            "target omzet",
            "product knowledge",
            "presentasi",
            "bahasa Inggris komunikasi",
            "retensi pelanggan",
            "Moka POS",
            "WhatsApp Business",
        ],
        "context": (
            "Konteks Indonesia: mall, dealer otomotif, call center, "
            "franchise F&B, marketplace seller, bank syariah front office."
        ),
    },
}


def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY tidak ditemukan. "
            "Set via: export GEMINI_API_KEY='...' "
            "atau isi file .env di root proyek."
        )
    return api_key.strip()


def build_system_instruction() -> str:
    return (
        "Anda adalah generator dataset NER untuk CV bahasa Indonesia.\n"
        "OUTPUT: satu objek JSON valid saja, tanpa markdown, tanpa penjelasan.\n"
        "ENTITAS: SKILL (keahlian), ROLE (jabatan).\n"
        "LABEL IOB: O, B-SKILL, I-SKILL, B-ROLE, I-ROLE.\n"
        "ATURAN: len(tokens) == len(labels); tokenisasi per spasi.\n"
        "Konteks: pasar kerja Indonesia (lokasi, budaya kerja, tools yang umum dipakai).\n"
    )


def setup_gemini() -> genai.GenerativeModel:
    api_key = get_gemini_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": GENERATION_TEMPERATURE,
            "max_output_tokens": 8192,
        },
    )


def cv_id_for(sector: str, index: int) -> str:
    prefix = SECTOR_ID_PREFIX[sector]
    return f"{prefix}_{index:04d}"


def build_variation_context(sector: str, index: int) -> str:
    profile = EXPERIENCE_LEVELS[index % len(EXPERIENCE_LEVELS)]
    city = INDONESIAN_CITIES[(index * 3) % len(INDONESIAN_CITIES)]
    company = COMPANY_CONTEXTS[(index * 7) % len(COMPANY_CONTEXTS)]
    realism = SECTOR_REALISM[sector]
    roles = realism["roles"]
    skills = realism["skills"]
    role_hint = roles[index % len(roles)]
    skill_a = skills[index % len(skills)]
    skill_b = skills[(index + 5) % len(skills)]
    skill_c = skills[(index + 11) % len(skills)]

    return (
        f"PROFIL VARIASI (wajib dipatuhi agar unik):\n"
        f"- Serial dataset: {index} (setiap CV harus berbeda dari serial lain)\n"
        f"- Level karier: {profile[0]} ({profile[1]})\n"
        f"- Narasi: {profile[2]}\n"
        f"- Kota relevan: {city}\n"
        f"- Lingkungan kerja: {company}\n"
        f"- Saran peran utama (boleh dimodifikasi): {role_hint}\n"
        f"- Saran skill (campur, jangan copy semua): {skill_a}, {skill_b}, {skill_c}\n"
        f"- {realism['context']}\n"
        f"WAJIB UNIK: jangan gunakan kalimat pembuka atau struktur yang sama "
        f"dengan CV lain. Variasikan peran, skill, pengalaman, dan tujuan karier.\n"
    )


def build_full_prompt(
    sector: str,
    cv_id: str,
    index: int,
    duplicate_feedback: str | None = None,
) -> str:
    variation = build_variation_context(sector, index)
    duplicate_block = ""
    if duplicate_feedback:
        duplicate_block = (
            f"\nPERINGATAN DUPLIKAT:\n{duplicate_feedback}\n"
            "Tulis CV baru yang benar-benar berbeda (peran, skill, kalimat).\n"
        )

    return (
        f"{build_system_instruction()}\n\n"
        f"TUGAS:\n"
        f"Buat 1 CV fiktif untuk sektor: {sector}\n"
        f"id: {cv_id}\n"
        f"language: id\n"
        f"Panjang raw_text: 90-160 kata.\n"
        f"Minimal {MIN_B_SKILL} entitas SKILL (B-SKILL) dan {MIN_B_ROLE} entitas ROLE (B-ROLE).\n"
        f"Gunakan Bahasa Indonesia natural (boleh istilah Inggris untuk skill teknis).\n"
        f"Field JSON wajib: id, sector, language, raw_text, tokens, labels\n"
        f"sector di JSON harus persis: {sector}\n"
        f"{variation}\n"
        f"{duplicate_block}"
        f'Contoh IOB: tokens ["Posisi","Backend","Engineer"] '
        f'labels ["O","B-ROLE","I-ROLE"]\n'
        f"\nATURAN KETAT JSON:\n"
        f"- raw_text: satu baris, TANPA tanda kutip ganda (\") di dalam teks.\n"
        f"- tokens: hasil pecah raw_text pada spasi (satu token = satu kata).\n"
        f"- labels: SATU label untuk SETIAP token, jumlah labels WAJIB = jumlah tokens.\n"
        f"- Sebelum mengirim, hitung: len(tokens) == len(labels).\n"
    )


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def content_fingerprint(raw_text: str) -> str:
    normalized = " ".join(raw_text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fingerprint_path_for_sector(sector: str) -> Path:
    prefix = SECTOR_ID_PREFIX[sector]
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)
    return FINGERPRINT_DIR / f"{prefix}_fingerprints.txt"


def load_sector_fingerprints(sector: str) -> set[str]:
    path = fingerprint_path_for_sector(sector)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def register_fingerprint(sector: str, fingerprint: str) -> None:
    path = fingerprint_path_for_sector(sector)
    with path.open("a", encoding="utf-8") as f:
        f.write(fingerprint + "\n")


def rebuild_fingerprints_from_disk(sector: str) -> set[str]:
    prefix = SECTOR_ID_PREFIX[sector]
    fps: set[str] = set()
    for path in OUTPUT_DIR.glob(f"{prefix}_*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                record = json.load(f)
            fps.add(content_fingerprint(str(record.get("raw_text", ""))))
        except (json.JSONDecodeError, OSError):
            continue
    fp_path = fingerprint_path_for_sector(sector)
    fp_path.write_text("\n".join(sorted(fps)) + ("\n" if fps else ""), encoding="utf-8")
    return fps


def normalize_cv_record(record: dict) -> dict:
    raw_text = str(record.get("raw_text", "")).strip()
    tokens = raw_text.split()
    labels = list(record.get("labels", []))
    delta = abs(len(tokens) - len(labels))

    if len(tokens) != len(labels):
        if delta > MAX_LABEL_PAD:
            raise ValueError(
                f"Selisih tokens/labels terlalu besar: tokens={len(tokens)}, "
                f"labels={len(labels)} (delta={delta})"
            )
        if len(labels) < len(tokens):
            labels = labels + ["O"] * (len(tokens) - len(labels))
        else:
            labels = labels[: len(tokens)]

    record["raw_text"] = raw_text
    record["tokens"] = tokens
    record["labels"] = labels
    return record


def enforce_record_metadata(record: dict, sector: str, cv_id: str) -> dict:
    record["id"] = cv_id
    record["sector"] = sector
    record["language"] = "id"
    return record


def validate_cv_record(record: dict) -> None:
    required = {"id", "sector", "language", "raw_text", "tokens", "labels"}
    if not required.issubset(record.keys()):
        raise ValueError(f"Field hilang: {required - set(record.keys())}")

    tokens = record["tokens"]
    labels = record["labels"]

    if not isinstance(tokens, list) or not isinstance(labels, list):
        raise ValueError("tokens/labels harus list")

    if len(tokens) != len(labels):
        raise ValueError(
            f"Panjang tidak sama: tokens={len(tokens)}, labels={len(labels)}"
        )

    if len(tokens) < MIN_TOKENS:
        raise ValueError(f"CV terlalu pendek: {len(tokens)} token (min {MIN_TOKENS})")

    if any(lab not in ALLOWED_LABELS for lab in labels):
        raise ValueError("Ada label di luar IOB yang diizinkan")

    b_skill = sum(1 for lab in labels if lab == "B-SKILL")
    b_role = sum(1 for lab in labels if lab == "B-ROLE")
    if b_skill < MIN_B_SKILL:
        raise ValueError(f"Terlalu sedikit B-SKILL: {b_skill} (min {MIN_B_SKILL})")
    if b_role < MIN_B_ROLE:
        raise ValueError(f"Terlalu sedikit B-ROLE: {b_role} (min {MIN_B_ROLE})")

    for i, lab in enumerate(labels):
        if lab.startswith("I-"):
            ent = lab[2:]
            if i == 0:
                raise ValueError("I- di posisi awal")
            prev = labels[i - 1]
            if prev not in {f"B-{ent}", f"I-{ent}"}:
                raise ValueError(f"IOB invalid: {lab} setelah {prev}")


def log_failed(cv_id: str, error: Exception) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {cv_id} | {error}\n")


def save_progress(progress: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {"total_saved": 0, "sectors": {}}
    with PROGRESS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def generate_one_cv(
    model: genai.GenerativeModel,
    sector: str,
    cv_id: str,
    index: int,
    known_fingerprints: set[str],
) -> dict:
    last_error: Exception | None = None

    for dup_attempt in range(1, MAX_DUPLICATE_RETRIES + 1):
        duplicate_feedback = None
        if dup_attempt > 1 and last_error:
            duplicate_feedback = str(last_error)

        prompt = build_full_prompt(
            sector=sector,
            cv_id=cv_id,
            index=index,
            duplicate_feedback=duplicate_feedback,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                current_prompt = prompt
                if attempt > 1 and last_error:
                    current_prompt += (
                        f"\n\nPERBAIKAN: Percobaan sebelumnya gagal karena: {last_error}\n"
                        "Pastikan JSON valid dan len(tokens) == len(labels)."
                    )
                response = model.generate_content(current_prompt)
                record = parse_json_response(response.text)
                record = normalize_cv_record(record)
                record = enforce_record_metadata(record, sector=sector, cv_id=cv_id)
                validate_cv_record(record)

                fp = content_fingerprint(record["raw_text"])
                if fp in known_fingerprints:
                    raise ValueError(
                        "CV duplikat: raw_text sama atau hampir sama dengan CV lain di sektor ini"
                    )

                known_fingerprints.add(fp)
                return record

            except json.JSONDecodeError as exc:
                last_error = exc
                debug_path = OUTPUT_DIR / f"debug_{cv_id}_attempt{attempt}.txt"
                debug_path.write_text(response.text, encoding="utf-8")
                print(f"  [retry {attempt}/{MAX_RETRIES}] gagal: {exc}")
                time.sleep(2)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    print(f"  [retry {attempt}/{MAX_RETRIES}] gagal: {exc}")
                time.sleep(2)

        if dup_attempt < MAX_DUPLICATE_RETRIES:
            print(f"  [dup retry {dup_attempt}/{MAX_DUPLICATE_RETRIES}] regenerasi karena: {last_error}")

    raise RuntimeError(f"Gagal generate {cv_id}: {last_error}")


def save_cv_record(
    record: dict,
    sector: str,
    fingerprint: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    register_fingerprint(sector, fingerprint)
    return path


def count_existing_for_sector(sector: str) -> int:
    prefix = SECTOR_ID_PREFIX[sector]
    return len(list(OUTPUT_DIR.glob(f"{prefix}_*.json")))


def run_sector_batch(
    model: genai.GenerativeModel,
    sector: str,
    start_index: int,
    end_index: int,
    progress: dict,
) -> int:
    known_fingerprints = load_sector_fingerprints(sector)
    if not known_fingerprints:
        known_fingerprints = rebuild_fingerprints_from_disk(sector)

    saved_this_run = 0
    prefix = SECTOR_ID_PREFIX[sector]

    for index in range(start_index, end_index + 1):
        cv_id = cv_id_for(sector, index)
        out_path = OUTPUT_DIR / f"{cv_id}.json"

        if out_path.exists():
            continue

        total_target = CVS_PER_SECTOR * len(PILOT_SECTORS)
        done_global = progress.get("total_saved", 0)
        print(
            f"[{sector}] {cv_id} ({index}/{end_index}) | "
            f"global ~{done_global}/{total_target} | fps={len(known_fingerprints)}"
        )

        try:
            record = generate_one_cv(
                model,
                sector=sector,
                cv_id=cv_id,
                index=index,
                known_fingerprints=known_fingerprints,
            )
            fp = content_fingerprint(record["raw_text"])
            path = save_cv_record(record, sector=sector, fingerprint=fp)
            saved_this_run += 1
            progress["total_saved"] = progress.get("total_saved", 0) + 1
            progress.setdefault("sectors", {})[sector] = count_existing_for_sector(sector)
            progress["last_cv_id"] = cv_id
            progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_progress(progress)
            print(
                f"  -> OK {path.name} | tokens={len(record['tokens'])} | "
                f"B-SKILL={sum(1 for l in record['labels'] if l == 'B-SKILL')}"
            )
        except Exception as exc:
            log_failed(cv_id, exc)
            print(f"  -> GAGAL {cv_id}: {exc}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return saved_this_run


def run_pilot_5_sectors(model: genai.GenerativeModel) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for i, sector in enumerate(PILOT_SECTORS):
        cv_id = cv_id_for(sector, index=1)
        out_path = OUTPUT_DIR / f"{cv_id}.json"

        if out_path.exists():
            print(f"[{i + 1}/5] Skip (sudah ada): {cv_id} ({sector})")
            saved_paths.append(out_path)
            continue

        print(f"[{i + 1}/5] Generating: {cv_id} ({sector})...")
        fps = load_sector_fingerprints(sector) or rebuild_fingerprints_from_disk(sector)
        record = generate_one_cv(
            model, sector=sector, cv_id=cv_id, index=1, known_fingerprints=fps
        )
        fp = content_fingerprint(record["raw_text"])
        path = save_cv_record(record, sector=sector, fingerprint=fp)
        saved_paths.append(path)
        print(
            f"  -> tersimpan: {path.name} | "
            f"tokens={len(record['tokens'])} labels={len(record['labels'])}"
        )

        if i < len(PILOT_SECTORS) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return saved_paths


def run_full_scale(
    model: genai.GenerativeModel,
    sectors: list[str],
    start_index: int,
    end_index: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    print(
        f"===== MAPAN CV Generator — FULL SCALE =====\n"
        f"Target: {CVS_PER_SECTOR} CV/sektor x {len(sectors)} sektor "
        f"= {CVS_PER_SECTOR * len(sectors)} total\n"
        f"Indeks per sektor: {start_index}..{end_index}\n"
    )

    for sector in sectors:
        existing = count_existing_for_sector(sector)
        print(f"\n--- Sektor: {sector} (sudah ada: {existing}/{CVS_PER_SECTOR}) ---")
        saved = run_sector_batch(
            model,
            sector=sector,
            start_index=start_index,
            end_index=end_index,
            progress=progress,
        )
        print(f"Sektor {sector}: +{saved} file baru pada run ini.")

    print("\n===== Ringkasan akhir =====")
    total = 0
    for sector in PILOT_SECTORS:
        n = count_existing_for_sector(sector)
        total += n
        print(f"  {sector}: {n}/{CVS_PER_SECTOR}")
    print(f"  TOTAL: {total}/{CVS_PER_SECTOR * len(PILOT_SECTORS)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPAN synthetic CV generator")
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="full",
        help="pilot = 1 CV per sektor; full = batch 600 per sektor",
    )
    parser.add_argument(
        "--sector",
        choices=PILOT_SECTORS,
        default=None,
        help="Hanya jalankan satu sektor (mode full)",
    )
    parser.add_argument("--start", type=int, default=1, help="Indeks CV awal (1-based)")
    parser.add_argument(
        "--end",
        type=int,
        default=CVS_PER_SECTOR,
        help=f"Indeks CV akhir (default {CVS_PER_SECTOR})",
    )
    parser.add_argument(
        "--rebuild-fingerprints",
        action="store_true",
        help="Rebuild file fingerprint dari JSON yang sudah ada, lalu keluar",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.rebuild_fingerprints:
        for sector in PILOT_SECTORS:
            fps = rebuild_fingerprints_from_disk(sector)
            print(f"{sector}: {len(fps)} fingerprint")
        sys.exit(0)

    if args.start < 1 or args.end < args.start:
        raise SystemExit("--start harus >= 1 dan --end >= --start")

    model = setup_gemini()

    if args.mode == "pilot":
        paths = run_pilot_5_sectors(model)
        print(f"\nSelesai pilot. Total file: {len(paths)}")
    else:
        sectors = [args.sector] if args.sector else list(PILOT_SECTORS)
        run_full_scale(
            model,
            sectors=sectors,
            start_index=args.start,
            end_index=min(args.end, CVS_PER_SECTOR),
        )
