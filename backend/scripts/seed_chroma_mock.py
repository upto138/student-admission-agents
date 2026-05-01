"""
Seed mock data into ChromaDB for local development/testing.
Includes sample admission data from popular universities.

Usage:
    RAG_BACKEND=chroma python scripts/seed_chroma_mock.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.tools.rag_tool import add_to_knowledge_base

MOCK_DATA = [
    {
        "content": (
            "Đại học Bách Khoa TP.HCM (HCMUT) tuyển sinh năm 2025. "
            "Ngành Khoa học Máy tính: điểm chuẩn 26.5 (thang 30), phương thức THPTQG. "
            "Ngành Kỹ thuật Phần mềm: điểm chuẩn 27.0. "
            "Học phí: 25-30 triệu/năm. "
            "Hạn nộp hồ sơ xét tuyển bổ sung: 30/08/2025."
        ),
        "source_url": "https://hcmut.edu.vn/tuyensinh/2025",
        "university": "HCMUT",
        "year": "2025",
    },
    {
        "content": (
            "Đại học Khoa học Tự nhiên TP.HCM (HCMUS) tuyển sinh 2025. "
            "Ngành Công nghệ Thông tin: điểm chuẩn 24.0 (THPTQG), 850 điểm (ĐGNL ĐHQG). "
            "Ngành Toán - Tin học: 22.5 điểm. "
            "Học phí chương trình chuẩn: 15 triệu/năm. "
            "Website: tuyensinh.hcmus.edu.vn"
        ),
        "source_url": "https://tuyensinh.hcmus.edu.vn/2025",
        "university": "HCMUS",
        "year": "2025",
    },
    {
        "content": (
            "Phương thức xét tuyển đại học 2025 phổ biến tại Việt Nam gồm: "
            "1. THPTQG - Xét điểm thi tốt nghiệp THPT (thang 30). "
            "2. ĐGNL - Đánh giá năng lực của ĐHQG (thang 1200). "
            "3. Học bạ THPT - Xét điểm trung bình các môn. "
            "Thời gian đăng ký xét tuyển trực tuyến: 01/07 - 30/07/2025 trên hệ thống tuyensinh.vnies.edu.vn."
        ),
        "source_url": "https://moet.gov.vn/tuyen-sinh-2025",
        "university": "",
        "year": "2025",
    },
]

def main():
    print("Seeding mock data into ChromaDB...")  # Start seeding process
    for i, item in enumerate(MOCK_DATA, 1):
        result = add_to_knowledge_base(**item)
        print(f"  [{i}/{len(MOCK_DATA)}] {item['university'] or 'General'}: {result}")  # Log each item
    print("Done.")  # Seeding complete

if __name__ == "__main__":
    main()