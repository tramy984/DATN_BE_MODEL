import re
import pandas as pd
from typing import Set

from tqdm import tqdm


class SkillExtractor:
    """Extract skills from multi-industry job descriptions"""

    def __init__(self):
        self.skill_alias = {
            'edit ảnh': 'chỉnh sửa ảnh',
            'edit video': 'chỉnh sửa video',
            'mc': 'dẫn chương trình',
            'hsk': 'tiếng trung',
            'n3': 'tiếng nhật',
            'korean': 'tiếng hàn',
            'jlpt': 'tiếng nhật',
            'n1': 'tiếng nhật',
            'n2': 'tiếng nhật',
            'n4': 'tiếng nhật',
            'n5': 'tiếng nhật',
            'topik': 'tiếng hàn',
            'ml': 'machine learning',
            'ai': 'machine learning',
            'reactjs': 'react',
            'react.js': 'react',
            'business analyst': 'ba',
            'node': 'nodejs',
            'node.js': 'nodejs',
            'vuejs': 'vue',
            'js': 'javascript',
            'ts': 'typescript',
            'restful api': 'rest api',
            'fb ads': 'facebook ads',
            'gg ads': 'google ads',
            'chăm sóc khách hàng': 'customer service',
            'cskh': 'customer service',
            'tiếng anh': 'english',
            'toeic': 'english',
            'ielts': 'english',
            'b1': 'english',
            'b2': 'english',
            'c1': 'english',
            'c2': 'english',
            'khả năng lãnh đạo': 'leadership',
            'kĩ năng lãnh đạo': 'leadership',
            'ui/ux': 'ui ux',
            'ux/ui': 'ui ux',
            'tuyển dụng': 'recruitment',
            'accounting': 'kế toán',
            'xnk': 'xuất nhập khẩu',
        }
        # =============================
        # IT / TECH
        # =============================
        self.tech_skills = {
            'python',
            'java',
            'javascript',
            'typescript',
            'c++',
            'c#',
            'php',
            'go',
            'ruby',
            'swift',
            'kotlin',
            'scala',
            'sql server',
            'html',
            'css',
            'react',
            'angular',
            'vue',
            'nodejs',
            'django',
            'flask',
            'spring',
            'laravel',
            'tensorflow',
            'pytorch',
            'keras',
            'pandas',
            'numpy',
            'mysql',
            'postgresql',
            'mongodb',
            'postman',
            'selenium',
            'redis',
            'sqlite',
            'docker',
            'kubernetes',
            'aws',
            'linux',
            'git',
            'rest api',
            'graphql',
            'android',
            'ios',
            'flutter',
            'microservices',
            'api',
            'backend',
            'frontend',
            'fullstack',
            'devops',
            'cloud',
            'firebase',
            'machine learning',
            'deep learning',
            'nlp',
            'computer vision',
            'dashboard',
            'llm',
            'ba',
            'testcase',
            'automation testing',
            'manual testing',
            'performance testing',
            'security testing',
            'agile',
            'scrum',
            'blockchain',
            'web3',
            'smart contract',
            'solidity',
            'truffle',
            'hardhat',
            'an ninh mạng',
            'an ninh thông tin',
            'an toàn thông tin',
            'cyber security',
            'antivirus',
            'red team',
            'blue team',
            'wireshark',
            'firewall',
            'ids',
            'ips',
            'siem',
            'soc',
            'vpn',
            'ssl',
            'tls',
            'azure',
            'gcp',
            'quản trị mạng',

        }

        # =============================
        # BUSINESS / SALES
        # =============================
        self.business_skills = {
            'trực hotline',
            'call center',
            'tiếp nhận cuộc gọi',
            'tư vấn khách hàng',
            'phân tích thị trường',
            'nghiên cứu thị trường',
            'bán hàng',
            'kinh doanh',
            'xử lý khiếu nại',
            'tư vấn bảo hiểm',
            'tìm kiếm khách hàng',
            'thẩm định hồ sơ',
            'tư vấn bất động sản',
            'word',
            'phân tích đầu tư',
            'thẩm định giá',
            'powerpoint',
            'luật đất đai',
            'crm',
            'account management',
            'lead generation',
            'b2b',
            'telesales',
            'retail sales',
            'pipeline management',
            'chăm sóc khách hàng',
        }

        # =============================
        # MARKETING
        # =============================
        self.marketing_skills = {
            'digital marketing',
            'seo',
            'sem',
            'google ads',
            'facebook ads',
            'tiktok ads',
            'content marketing',
            'storytelling',
            'content creator',
            'copywriting',
            'social media',
            'email marketing',
            'branding',
            'performance marketing',
            'marketing online',
            'media buying',
            'google analytics',
            'ecommerce',
        }

        # =============================
        # FINANCE / ACCOUNTING/ PHÁP LÝ/HÀNH CHÍNH
        # =============================
        self.finance_skills = {
            'kế toán tổng hợp',
            'kế toán thuế',
            'kế toán nội bộ',
            'kế toán kho',
            'kế toán công nợ',
            'kế toán bán hàng',
            'hạch toán',
            'quyết toán thuế',
            'kê khai thuế',
            'quản lý hóa đơn',
            'quản lý rủi ro',
            'quản lý vận hành',
            'quản lý sản xuất',
            'quản lý kho',
            'quản lý sự kiện',
            'kiểm toán',
            'audit',
            'erp',
            'sap',
            'xử lý nợ',
            'soạn thảo hợp đồng',
            'tư vấn pháp lý',
            'tố tụng',
            'thủ tục pháp lý',
            'luật đầu tư',
            'luật doanh nghiệp',
            'luật lao động',
            'luật thương mại',
            'luật đất đai',
            'quản lý hồ sơ pháp lý',
            'kiểm tra hợp đồng',
            'hành chính văn phòng',
            'xử lý công văn',
            'sắp xếp lịch làm việc',
            'báo cáo tài chính',
            'cost accounting',

        }

        # =============================
        # HR
        # =============================
        self.hr_skills = {
            'quản lý nhân sự',
            'tuyển dụng',
            'quản lý hợp đồng lao động',
            'xây dựng thương hiệu tuyển dụng',
            'quản lý dự án',
            'đánh giá ứng viên',
            'hrm',
            'payroll',
            'c&b',
        }

        # =============================
        # ENGINEERING
        # =============================
        self.engineering_skills = {
            'thêu',
            'cắt vải',
            'may công nghiệp',
            'thiết kế rập',
            'xu hướng thời trang',
            'thiết kế mẫu thời trang',
            'vắt sổ',
            '1 kim',
            '2 kim',
            '3 kim',
            '4 kim',
            'thiết kế nội thất',
            'cơ khí',
            'cnc',
            'tiện',
            'phay',
            'autocad',
            'solidworks',
            'catia',
            'creo',
            'cad/cam',
            'thiết kế kiến trúc',
            'sketchup',
            'thiết kế 2d',
            'điện công nghiệp',
            'điện dân dụng',
            'plc',
            'scada',
            'hmi',
            'servo',
            'inverter',
            'tự động hóa',
            'robot',
            'sensor',
            'hvac',
            'chiller',
            'sửa chữa động cơ',
            'bảo trì máy móc',
            'bảo trì điện',
            'sửa chữa điện',
            'vận hành máy móc',
            'eplan',
            'qa',
            'qc',
            'iso',
            '5s',
            'six sigma',
            'kaizen',
            'cắt may',
            'sửa chữa máy móc',
            'inventor',
            'cam',
        }

        # =============================
        # LOGISTICS
        # =============================
        self.logistics_skills = {
            'kiểm tra chứng từ',
            'điều hành tour',
            'bán tour',
            'hướng dẫn viên',
            'lữ hành',
            'dẫn chương trình',
            'supply chain',

        }

        # =============================
        # DESIGN
        # =============================
        self.design_skills = {
            'photoshop',
            'figma',
            'after effects',
            'ui/ux',
            'graphic design',
            'motion graphic',
            '2d/3d design',
            '3ds max',
            'excel',
            'thiết kế đồ họa',
            'thiết kế giao diện',
            'phối cảnh',
            'illustrator',
            'wireframe',
            'giao dịch viên',
            'thẩm định tín dụng',
            'huy động vốn',
            'xử lý giao dịch',
            'ngân hàng điện tử',

        }

        # =============================
        # ART / CREATIVE
        # =============================
        self.art_skills = {
            '3d modeling',
            'animation',
            'motion graphic',
            'layout design',
            'typography',
        }

        # =============================
        # MEDIA / PR
        # =============================
        self.media_skills = {
            'copywriting',
            'pr',
            'media buying',
            'livestream',
            'journalism',
            'biên tập',
            'quay dựng',
            'chỉnh sửa ảnh',
            'chỉnh sửa video',
            'capcut',
            'capa',
            'fmea',
            'content creation',
            'public relations',
            'video editing',
            'premiere',
        }

        # =============================
        # HEALTHCARE + SPA + CÔNG NGHỆ SINH HỌC + DINH DƯỠNG +CNTP
        # =============================
        self.health_skills = {
            'vi sinh',
            'hóa sinh',
            'miễn dịch học',
            'điều dưỡng',
            'tiêm',
            'truyền dịch',
            'chăm sóc bệnh nhân',
            'xét nghiệm',
            'siêu âm',
            'x quang',
            'dược lâm sàng',
            'chăm sóc da',
            'soi da',
            'lấy mụn',
            'phi kim',
            'lăn kim',
            'peel da',
            'laser',
            'tiêm filler',
            'botox',
            'mesotherapy',
            'triệt lông',
            'phun xăm',
            'phun môi',
            'phun mày',
            'nối mi',
            'massage',
            'skin care',
            'nghiên cứu công thức',
            'phân tích mẫu',
            'sử dụng thiết bị phòng thí nghiệm',
            'pif',
            'nuôi cấy tế bào',
            'cấy mô',
            'di truyền học',
            'sinh học phân tử',
            'sinh học tế bào',
            'nuôi cấy vi sinh vật',
            'kỹ thuật di truyền',
            'gmp',
            'ssops',
            'haccp',
            'glp',
            'sop',
            'kiểm nghiệm thực phẩm',
            'hóa phân tích',
            'hóa hữu cơ',
            'hóa vô cơ',
            'hóa polymer',
            'phân tích dinh dưỡng',
            'kỹ thuật protein',
            'kỹ thuật mỏ',
            'kỹ thuật nuôi cấy tế bào',
            'kỹ thuật nuôi cấy vi sinh vật',
        }

        # =============================
        # EDUCATION
        # =============================
        self.education_skills = {
            'giảng dạy',
            'đứng lớp',
            'nghiệp vụ sư phạm',
            'quản lý lớp học',
            'soạn giáo trình',
            'soạn giáo án',
            'thiết kế bài giảng',
            'thiết kế slide',
        }

        # =============================
        # CONSTRUCTION
        # =============================
        self.construction_skills = {
            'kỹ sư xây dựng',
            'revit',
            'autocad',
            'dự toán công trình',
            'bóc tách khối lượng',
            'giám sát công trình',
            'site engineer',
            'shop drawing',
            'quản lý công trình',
            'safe',
            'bim',
            'sap2000',
            'etabs',
        }

        # =============================
        # CUSTOMER SERVICE
        # =============================
        self.cs_skills = {
            'call center',
            'telesales',
        }
        # =============================
        # FOREIGN LANGUAGE
        # =============================
        self.language_skills = {
            'biên dịch',
            'phiên dịch',
            'dịch tài liệu',
            'dịch thuật',
            'thông dịch',
            'tiếng trung',
            'tiếng nhật',
            'tiếng hàn',
            'tiếng anh',
            'tiếng pháp',
        }

        self.restaurant_skills = {
            'phục vụ bàn',
            'quản lý ca',
            'phụ bếp',
            'quản lý nhà hàng',
            'quản lý khách sạn',
            'quản lý phòng',
            'quản lý bếp',
            'pha chế',
            'lễ tân',
        }
        self.manage_skills = {
            'tối ưu hoạt động kỹ thuật',
        }
        self.product_skills = {
            'lean',
        }

        self.aquatic_skills = {
            'kiểm soát chất lượng',
        }
        self.food_skills = {
            'pha chế',
        }
        # =============================
        # DEGREE
        # =============================
        self.degree_keywords = {
            "trung cấp": "degree_trung_cap",
            "cao đẳng": "degree_cao_dang",
            "đại học": "degree_dai_hoc",
            "cử nhân": "degree_cu_nhan",
            "bachelor": "degree_cu_nhan",
            "thạc sĩ": "degree_thac_si",
            "master": "degree_thac_si",
            "tiến sĩ": "degree_tien_si",
            "phd": "degree_tien_si",
        }

        self.all_skills = (
              self.tech_skills
            | self.business_skills
            | self.marketing_skills
            | self.finance_skills
            | self.hr_skills
            | self.engineering_skills
            | self.logistics_skills
            | self.design_skills
            | self.art_skills
            | self.media_skills
            | self.health_skills
            | self.education_skills
            | self.construction_skills
            | self.cs_skills
            | self.language_skills
            | self.restaurant_skills
            | self.manage_skills
            | self.product_skills
            | self.aquatic_skills
            | self.food_skills
             

        )
    def normalize_skill(self, skill):
        skill = skill.lower().strip()
        return self.skill_alias.get(skill, skill)
    def extract_skills(self, text: str) -> Set[str]:

        if not text:
            return set()

        text = text.lower()
        found_skills = set()

        # detect alias FIRST
        for alias, canonical in self.skill_alias.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                found_skills.add(canonical)

        # detect skills
        for skill in self.all_skills:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text):
                found_skills.add(self.normalize_skill(skill))

        # degree
        for k, v in self.degree_keywords.items():
            if k in text:
                found_skills.add(v)

        return found_skills

    def _extract_years_experience(self, skills: Set[str]) -> int:
        for skill in skills:
            if "_years_exp" in skill:
                try:
                    return int(skill.split("_")[0])
                except:
                    pass
        return 0
    