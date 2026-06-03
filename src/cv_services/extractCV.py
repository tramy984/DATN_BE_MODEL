import re
import json
import pdfplumber
from unidecode import unidecode

from cv_services.skill_extractor import SkillExtractor


# =========================
# 1. LOAD PDF
# =========================

def extract_text_from_pdf(pdf_path):
    text = ""

    if hasattr(pdf_path, "seek"):
        pdf_path.seek(0)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text(
                x_tolerance=1,
                y_tolerance=3,
                layout=True
            )

            if page_text:
                text += f"\n===== PAGE {i + 1} =====\n"
                text += page_text + "\n"

    return text.strip()


# =========================
# 2. LOAD INDUSTRY JSON
# =========================

def load_industry_skill_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 3. TITLE KEYWORDS
# =========================

INDUSTRY_TITLE_KEYWORDS = {
    "CNTT - Phần mềm": [
        "developer", "software", "backend", "frontend", "fullstack",
        "programmer", "lập trình", "software development",
        "web developer", "data analyst", "data scientist", "ai engineer"
    ],

    "CNTT - Phần cứng / Mạng": [
        "network", "security", "cyber", "it support",
        "quản trị mạng", "an ninh mạng"
    ],

    "Bán hàng / Kinh doanh": [
        "sales", "kinh doanh", "bán hàng",
        "business development", "account executive"
    ],

    "Bán lẻ / Bán sỉ": [
        "bán lẻ", "bán sỉ", "retail", "retail sales"
    ],

    "Bảo hiểm": [
        "bảo hiểm", "insurance"
    ],

    "Bất động sản": [
        "bất động sản", "real estate"
    ],

    "Biên phiên dịch": [
        "biên dịch", "phiên dịch", "translator", "interpreter"
    ],

    "Tiếp thị / Marketing": [
        "marketing", "digital marketing", "content", "seo", "social media"
    ],

    "Tiếp thị trực tuyến": [
        "digital marketing", "online marketing", "seo", "sem"
    ],

    "Thương mại điện tử": [
        "ecommerce", "e-commerce", "thương mại điện tử"
    ],

    "Kế toán / Kiểm toán": [
        "kế toán", "kiểm toán", "accountant", "accounting", "auditor"
    ],

    "Tài chính / Đầu tư": [
        "tài chính", "đầu tư", "finance", "investment"
    ],

    "Ngân hàng": [
        "ngân hàng", "banking", "giao dịch viên"
    ],

    "Nhân sự": [
        "nhân sự", "hr", "human resource", "recruitment", "tuyển dụng"
    ],

    "Hành chính / Thư ký": [
        "hành chính", "thư ký", "admin", "secretary"
    ],

    "Luật / Pháp lý": [
        "luật", "pháp lý", "legal", "lawyer"
    ],

    "Mỹ thuật / Nghệ thuật / Thiết kế": [
        "designer", "graphic", "ui ux", "thiết kế", "photoshop"
    ],

    "Kiến trúc": [
        "kiến trúc", "architect", "architecture"
    ],

    "Nội ngoại thất": [
        "nội thất", "ngoại thất", "interior"
    ],

    "Giáo dục / Đào tạo": [
        "giáo viên", "teacher", "trainer", "giảng viên", "giảng dạy"
    ],

    "Y tế / Chăm sóc sức khỏe / Thẩm mỹ / Làm đẹp": [
        "điều dưỡng", "y tá", "bác sĩ", "nurse", "doctor",
        "spa", "chăm sóc da", "thẩm mỹ"
    ],

    "Xây dựng": [
        "xây dựng", "site engineer", "kỹ sư xây dựng", "giám sát công trình"
    ],

    "Cơ khí / Ô tô / Tự động hóa": [
        "cơ khí", "ô tô", "tự động hóa", "mechanical", "automation"
    ],

    "Điện / Điện tử / Điện lạnh / Điện công nghiệp": [
        "điện", "điện tử", "điện lạnh", "electrical"
    ],

    "Nhà hàng / Khách sạn": [
        "lễ tân", "phục vụ", "nhà hàng", "khách sạn", "receptionist"
    ],

    "Du lịch": [
        "du lịch", "tour", "hướng dẫn viên"
    ],

    "Quản lý chất lượng (QA/QC)": [
        "qa", "qc", "quality control", "quality assurance"
    ],

    "Quản lý điều hành": [
        "quản lý điều hành", "operation manager", "operations"
    ],

    "Sản xuất / Vận hành sản xuất": [
        "sản xuất", "vận hành sản xuất", "production"
    ],

    "Hóa học": [
        "hóa học", "chemist", "chemical"
    ],

    "Công nghệ sinh học": [
        "công nghệ sinh học", "biotechnology"
    ],

    "Công nghệ thực phẩm / Dinh dưỡng": [
        "công nghệ thực phẩm", "dinh dưỡng", "food technology"
    ],

    "Dệt may / Da giày / Thời trang": [
        "dệt may", "thời trang", "fashion", "garment"
    ],

    "Bảo trì / Sửa chữa": [
        "bảo trì", "sửa chữa", "maintenance", "repair"
    ],

    "Truyền hình / Báo chí / Biên tập": [
        "báo chí", "biên tập", "journalist", "editor"
    ],

    "Quảng cáo / Đối ngoại / Truyền Thông": [
        "truyền thông", "pr", "public relations", "quảng cáo"
    ],
}


# =========================
# 4. INDUSTRY PREDICT HELPERS
# =========================

def extract_cv_title(raw_text):
    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    title_keywords = [
        "intern",
        "developer",
        "engineer",
        "accountant",
        "accounting",
        "finance",
        "financial",
        "investment",
        "banking",
        "auditor",
        "marketing",
        "sales",
        "hr",
        "recruitment",
        "designer",

        "thực tập",
        "nhân viên",
        "chuyên viên",
        "kế toán",
        "kiểm toán",
        "tài chính",
        "đầu tư",
        "ngân hàng",
        "kinh doanh",
        "marketing",
        "nhân sự",
        "thiết kế",
        "lập trình"
    ]

    ignore_keywords = [
        "@",
        "gmail",
        "email",
        "phone",
        "điện thoại",
        "github",
        "linkedin",
        "===== page"
    ]

    for line in lines[:25]:
        line_lower = line.lower()

        if any(k in line_lower for k in ignore_keywords):
            continue

        if any(k in line_lower for k in title_keywords):
            return line

    return ""


def build_skill_industry_frequency(industry_map):
    skill_count = {}

    for industry, data in industry_map.items():
        all_skills = (
            set(data.get("base_skills", []))
            | set(data.get("optional_skills", []))
        )

        for skill in all_skills:
            skill_count[skill] = skill_count.get(skill, 0) + 1

    return skill_count


def count_keyword_frequency(raw_text, keyword):
    text = unidecode(raw_text.lower())
    keyword_norm = unidecode(keyword.lower())

    pattern = r"\b" + re.escape(keyword_norm) + r"\b"

    return len(re.findall(pattern, text))


def calculate_title_score(industry, raw_text):
    title = extract_cv_title(raw_text)
    title_norm = unidecode(title.lower())

    score = 0
    matched_titles = []

    for keyword in INDUSTRY_TITLE_KEYWORDS.get(industry, []):
        keyword_norm = unidecode(keyword.lower())

        if keyword_norm in title_norm:
            score += 40
            matched_titles.append(keyword)

    return score, matched_titles, title


def predict_industry(skills, industry_map, raw_text=""):
    skill_set = set(skills)

    skill_industry_count = build_skill_industry_frequency(industry_map)

    weak_skills = {
        "excel",
        "word",
        "powerpoint",
        "english",
        "tiếng anh",
        "chăm sóc khách hàng",
        "customer service",
        "crm"
    }

    best_industry = "Ngành khác"
    best_score = 0
    best_matched = []
    top_scores = []

    for industry, data in industry_map.items():
        base_skills = set(data.get("base_skills", []))
        optional_skills = set(data.get("optional_skills", []))
        industry_skills = base_skills | optional_skills

        matched_base = skill_set & base_skills
        matched_optional = skill_set & optional_skills
        matched_all = matched_base | matched_optional

        # Tầng 1: title
        title_score, matched_titles, detected_title = calculate_title_score(
            industry,
            raw_text
        )

        # Tầng 2: skill đặc trưng
        skill_score = 0
        skill_details = []

        for skill in matched_all:
            appears_in_n_industries = skill_industry_count.get(skill, 1)
            uniqueness_weight = 1 / appears_in_n_industries

            if skill in base_skills:
                weight = 20 * uniqueness_weight
            else:
                weight = 8 * uniqueness_weight

            if skill in weak_skills:
                weight *= 0.2

            skill_score += weight

            skill_details.append({
                "skill": skill,
                "type": "base" if skill in base_skills else "optional",
                "appears_in_industries": appears_in_n_industries,
                "weight": round(weight, 4)
            })

        # Tầng 4: frequency
        frequency_score = 0

        for skill in industry_skills:
            freq = count_keyword_frequency(raw_text, skill)

            if freq > 0:
                appears_in_n_industries = skill_industry_count.get(skill, 1)
                freq_weight = 1 / appears_in_n_industries

                if skill in base_skills:
                    add_score = min(freq, 5) * 3 * freq_weight
                else:
                    add_score = min(freq, 5) * 1.2 * freq_weight

                if skill in weak_skills:
                    add_score *= 0.2

                frequency_score += add_score

        final_score = (
            title_score * 0.45
            + skill_score * 0.40
            + frequency_score * 0.15
        )

        top_scores.append({
            "industry": industry,
            "final_score": round(final_score, 4),
            "title_score": round(title_score, 4),
            "skill_score": round(skill_score, 4),
            "frequency_score": round(frequency_score, 4),
            "detected_title": detected_title,
            "matched_titles": matched_titles,
            "matched_base": sorted(list(matched_base)),
            "matched_optional": sorted(list(matched_optional)),
            "skill_details": sorted(
                skill_details,
                key=lambda x: x["weight"],
                reverse=True
            )
        })

        if final_score > best_score:
            best_score = final_score
            best_industry = industry
            best_matched = list(matched_all)

    top_scores = sorted(
        top_scores,
        key=lambda x: x["final_score"],
        reverse=True
    )

    if best_score < 1:
        return "Ngành khác", best_score, best_matched, top_scores[:5]

    return best_industry, best_score, best_matched, top_scores[:5]


# =========================
# 5. DEGREE / EXPERIENCE / LOCATION
# =========================

VIETNAM_LOCATIONS = [
    "Hà Nội",
    "Hồ Chí Minh",
    "Đà Nẵng",
    "Hải Phòng",
    "Cần Thơ",

    "An Giang",
    "Bà Rịa - Vũng Tàu",
    "Bắc Giang",
    "Bắc Kạn",
    "Bạc Liêu",
    "Bắc Ninh",
    "Bến Tre",
    "Bình Định",
    "Bình Dương",
    "Bình Phước",
    "Bình Thuận",

    "Cà Mau",
    "Cao Bằng",

    "Đắk Lắk",
    "Đắk Nông",
    "Điện Biên",
    "Đồng Nai",
    "Đồng Tháp",

    "Gia Lai",

    "Hà Giang",
    "Hà Nam",
    "Hà Tĩnh",
    "Hải Dương",
    "Hậu Giang",
    "Hòa Bình",
    "Hưng Yên",

    "Khánh Hòa",
    "Kiên Giang",

    "Lai Châu",
    "Lâm Đồng",
    "Lạng Sơn",
    "Lào Cai",
    "Long An",

    "Nam Định",
    "Nghệ An",
    "Ninh Bình",
    "Ninh Thuận",

    "Phú Thọ",
    "Phú Yên",

    "Quảng Bình",
    "Quảng Nam",
    "Quảng Ngãi",
    "Quảng Ninh",
    "Quảng Trị",

    "Sóc Trăng",
    "Sơn La",

    "Tây Ninh",
    "Thái Bình",
    "Thái Nguyên",
    "Thanh Hóa",
    "Thừa Thiên Huế",
    "Tiền Giang",
    "Trà Vinh",
    "Tuyên Quang",

    "Vĩnh Long",
    "Vĩnh Phúc",

    "Yên Bái"
]

LOCATION_ALIASES = {

    # Hồ Chí Minh
    "tp hcm": "Hồ Chí Minh",
    "tphcm": "Hồ Chí Minh",
    "tp.hcm": "Hồ Chí Minh",
    "ho chi minh": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh",
    "sai gon": "Hồ Chí Minh",
    "saigon": "Hồ Chí Minh",

    # Hà Nội
    "ha noi": "Hà Nội",
    "hanoi": "Hà Nội",

    # Đà Nẵng
    "da nang": "Đà Nẵng",
    "danang": "Đà Nẵng",

    # Huế
    "hue": "Thừa Thiên Huế",

    # BRVT
    "vung tau": "Bà Rịa - Vũng Tàu",
    "ba ria": "Bà Rịa - Vũng Tàu",
    "brvt": "Bà Rịa - Vũng Tàu",

    # Dak Lak
    "dak lak": "Đắk Lắk",
    "daklak": "Đắk Lắk",

    # Dak Nong
    "dak nong": "Đắk Nông",
    "daknong": "Đắk Nông",

    # Can Tho
    "can tho": "Cần Thơ",

    # Hai Phong
    "hai phong": "Hải Phòng",

    # Khanh Hoa
    "nha trang": "Khánh Hòa",

    # Lam Dong
    "da lat": "Lâm Đồng",
    "dalat": "Lâm Đồng",

    # Quang Nam
    "tam ky": "Quảng Nam",
    "hoi an": "Quảng Nam",

    # Quang Ngai
    "quang ngai city": "Quảng Ngãi",

    # Nghe An
    "vinh": "Nghệ An",

    # Thanh Hoa
    "thanh hoa city": "Thanh Hóa",

    # Kien Giang
    "phu quoc": "Kiên Giang",

    # Binh Dinh
    "quy nhon": "Bình Định",

    # Gia Lai
    "pleiku": "Gia Lai",

    # Lao Cai
    "sapa": "Lào Cai",
    "sa pa": "Lào Cai"
}

def find_city_in_text(text):
    if not text:
        return None

    text_norm = unidecode(text.lower())

    for alias, location in LOCATION_ALIASES.items():
        if alias in text_norm:
            return location

    for location in VIETNAM_LOCATIONS:
        location_norm = unidecode(location.lower())

        if location_norm in text_norm:
            return location

    return None


def extract_location(text):
    if not text:
        return None

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    location_keywords = [
        "địa chỉ",
        "address",
        "location",
        "nơi ở",
        "hiện tại",
        "sinh sống",
        "liên hệ"
    ]

    ignore_keywords = [
        "đại học",
        "university",
        "college",
        "trường",
        "education",
        "học vấn",
        "dự án",
        "project",
        "github",
        "website",
        "kí túc xá",
        "ký túc xá"
    ]

    for line in lines:
        line_lower = line.lower()

        if (
            "" in line
            or "📍" in line
            or any(keyword in line_lower for keyword in location_keywords)
        ):
            city = find_city_in_text(line)

            if city:
                return city

    for line in lines[:12]:
        line_lower = line.lower()

        if any(keyword in line_lower for keyword in ignore_keywords):
            continue

        city = find_city_in_text(line)

        if city:
            return city

    for line in lines:
        line_lower = line.lower()

        if any(keyword in line_lower for keyword in ignore_keywords):
            continue

        city = find_city_in_text(line)

        if city:
            return city

    return None


def extract_degree(text, extractor):
    if not text:
        return None

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    education_keywords = [
        "học vấn",
        "education",
        "trình độ học vấn",
        "bằng cấp",
        "qualification"
    ]

    degree_priority = [
        ("tiến sĩ", "degree_tien_si"),
        ("phd", "degree_tien_si"),
        ("thạc sĩ", "degree_thac_si"),
        ("master", "degree_thac_si"),
        ("cử nhân", "degree_cu_nhan"),
        ("bachelor", "degree_cu_nhan"),
        ("kỹ sư", "degree_dai_hoc"),
        ("engineer", "degree_dai_hoc"),
        ("đại học", "degree_dai_hoc"),
        ("university", "degree_dai_hoc"),
        ("cao đẳng", "degree_cao_dang"),
        ("college", "degree_cao_dang"),
        ("trung cấp", "degree_trung_cap"),
    ]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if any(keyword in line_lower for keyword in education_keywords):
            block = " ".join(lines[i:i + 8]).lower()

            for keyword, value in degree_priority:
                if keyword in block:
                    return value

    for key, value in extractor.degree_keywords.items():
        if key.lower() in text.lower():
            return value

    return None


def extract_experience(text):
    if not text:
        return 0, 0

    text = text.lower()

    patterns = [
        r'(\d+)\+?\s*năm\s+kinh\s+nghiệm',
        r'kinh\s+nghiệm\s*:?\s*(\d+)\+?\s*năm',
        r'(\d+)\+?\s*years?\s+experience',
        r'experience\s*:?\s*(\d+)\+?\s*years?',
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                years.append(int(match))
            except Exception:
                pass

    if not years:
        return 0, 0

    return min(years), max(years)


# =========================
# 6. BUILD OUTPUT
# =========================

def build_cv_text(industry, skills):
    clean_skills = []

    for skill in skills:
        if skill.startswith("degree_"):
            continue

        if skill.endswith("_years_exp"):
            continue

        clean_skills.append(skill)

    clean_skills = sorted(set(clean_skills))
    skill_text = ", ".join(clean_skills) if clean_skills else "Không xác định"

    return f"Candidate industry: {industry}. Skills: {skill_text}."


def extract_cv_profile(pdf_path, industry_json_path):
    extractor = SkillExtractor()
    industry_map = load_industry_skill_map(industry_json_path)

    raw_text = extract_text_from_pdf(pdf_path)

    skills = extractor.extract_skills(raw_text)

    industry, industry_score, matched_industry_skills, top_industries = predict_industry(
        skills,
        industry_map,
        raw_text
    )

    degree = extract_degree(raw_text, extractor)
    exp_min, exp_max = extract_experience(raw_text)
    location = extract_location(raw_text)
    cv_text = build_cv_text(industry, skills)

    return {
        "raw_text": raw_text,
        "skills": sorted(list(skills)),
        "industry": industry,
        "industry_score": industry_score,
        "matched_industry_skills": sorted(matched_industry_skills),
        "top_industries": top_industries,
        "degree": degree,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "location": location,
        "cv_text": cv_text
    }


if __name__ == "__main__":
    pdf_path = "ExtractCV/Nguyen-Thi-Tra-My-TopCV.vn-010626.202022.pdf"
    industry_json_path = "build_graph_main/skills_by_industry.json"

    result = extract_cv_profile(
        pdf_path,
        industry_json_path
    )

    print("\n" + "=" * 80)
    print("EXTRACTED SKILLS")
    print("=" * 80)
    print(result["skills"])

    print("\n" + "=" * 80)
    print("CANDIDATE INDUSTRY")
    print("=" * 80)
    print(result["industry"])
    print("Industry score:", result["industry_score"])
    print("Matched industry skills:", result["matched_industry_skills"])

    print("\n" + "=" * 80)
    print("TOP INDUSTRY CANDIDATES")
    print("=" * 80)

    for item in result["top_industries"]:
        print(item)

    print("\n" + "=" * 80)
    print("DEGREE / EXPERIENCE / LOCATION")
    print("=" * 80)
    print("Degree:", result["degree"])
    print("Exp min:", result["exp_min"])
    print("Exp max:", result["exp_max"])
    print("Location:", result["location"])

    print("\n" + "=" * 80)
    print("FINAL CV TEXT")
    print("=" * 80)
    print(result["cv_text"])
