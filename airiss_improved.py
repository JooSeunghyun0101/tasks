# AIRISS 완벽한 엔터프라이즈급 AI 분석 시스템 - AI 피드백 개선판
# AI 피드백 기능 완전 수정 및 개선
# 파일명: airiss_improved.py

import sys
import os
from pathlib import Path

# 필수 라이브러리 체크 및 자동 설치
def check_and_install_requirements():
    required_packages = [
        'fastapi',
        'uvicorn[standard]', 
        'pandas',
        'openpyxl',
        'python-multipart',
        'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'uvicorn[standard]':
                import uvicorn
            elif package == 'python-multipart':
                import multipart
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"🔧 누락된 패키지 설치 중: {', '.join(missing_packages)}")
        import subprocess
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ 모든 패키지 설치 완료!")

# 라이브러리 설치 확인
check_and_install_requirements()

# 이제 안전하게 import
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import io
import uuid
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import uvicorn
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="AIRISS Enterprise AI Analysis System",
    description="AI 기반 직원 성과/역량 종합 스코어링 시스템",
    version="3.1.0"
)

# 전역 저장소
class DataStore:
    def __init__(self):
        self.files = {}
        self.jobs = {}
        self.results = {}
    
    def add_file(self, file_id: str, data: Dict):
        self.files[file_id] = data
    
    def get_file(self, file_id: str) -> Optional[Dict]:
        return self.files.get(file_id)
    
    def add_job(self, job_id: str, data: Dict):
        self.jobs[job_id] = data
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)

store = DataStore()

# AIRISS 8대 영역 완전 설계
AIRISS_FRAMEWORK = {
    "업무성과": {
        "keywords": {
            "positive": [
                "우수", "탁월", "뛰어남", "성과", "달성", "완료", "성공", "효율", "생산적", 
                "목표달성", "초과달성", "품질", "정확", "신속", "완벽", "전문적", "체계적",
                "성과가", "결과를", "실적이", "완성도", "만족도"
            ],
            "negative": [
                "부족", "미흡", "지연", "실패", "문제", "오류", "늦음", "비효율", 
                "목표미달", "품질저하", "부정확", "미완성", "부실", "개선", "보완"
            ]
        },
        "weight": 0.25,
        "description": "업무 산출물의 양과 질",
        "color": "#FF6B6B"
    },
    "KPI달성": {
        "keywords": {
            "positive": [
                "KPI달성", "지표달성", "목표초과", "성과우수", "실적우수", "매출증가", 
                "효율향상", "생산성향상", "수치달성", "성장", "개선", "달성률", "초과"
            ],
            "negative": [
                "KPI미달", "목표미달", "실적부진", "매출감소", "효율저하", 
                "생산성저하", "수치부족", "하락", "퇴보", "미달"
            ]
        },
        "weight": 0.20,
        "description": "핵심성과지표 달성도",
        "color": "#4ECDC4"
    },
    "태도마인드": {
        "keywords": {
            "positive": [
                "적극적", "긍정적", "열정", "성실", "책임감", "진취적", "협조적", 
                "성장지향", "학습의지", "도전정신", "주인의식", "헌신", "열심히", "노력"
            ],
            "negative": [
                "소극적", "부정적", "무관심", "불성실", "회피", "냉소적", 
                "비협조적", "안주", "현상유지", "수동적", "태도", "마인드"
            ]
        },
        "weight": 0.15,
        "description": "업무에 대한 태도와 마인드셋",
        "color": "#45B7D1"
    },
    "커뮤니케이션": {
        "keywords": {
            "positive": [
                "명확", "정확", "신속", "친절", "경청", "소통", "전달", "이해", 
                "설득", "협의", "조율", "공유", "투명", "개방적", "의사소통", "원활"
            ],
            "negative": [
                "불명확", "지연", "무시", "오해", "단절", "침묵", "회피", 
                "독단", "일방적", "폐쇄적", "소통부족", "전달력"
            ]
        },
        "weight": 0.15,
        "description": "의사소통 능력과 스타일",
        "color": "#96CEB4"
    },
    "리더십협업": {
        "keywords": {
            "positive": [
                "리더십", "팀워크", "협업", "지원", "멘토링", "동기부여", "조율", 
                "화합", "팀빌딩", "위임", "코칭", "영향력", "협력", "팀플레이"
            ],
            "negative": [
                "독단", "갈등", "비협조", "소외", "분열", "대립", "이기주의", 
                "방해", "무관심", "고립", "개인주의"
            ]
        },
        "weight": 0.10,
        "description": "리더십과 협업 능력",
        "color": "#FFEAA7"
    },
    "전문성학습": {
        "keywords": {
            "positive": [
                "전문", "숙련", "기술", "지식", "학습", "발전", "역량", "능력", 
                "성장", "향상", "습득", "개발", "전문성", "노하우", "스킬", "경험"
            ],
            "negative": [
                "미숙", "부족", "낙후", "무지", "정체", "퇴보", "무능력", 
                "기초부족", "역량부족", "실력부족"
            ]
        },
        "weight": 0.08,
        "description": "전문성과 학습능력",
        "color": "#DDA0DD"
    },
    "창의혁신": {
        "keywords": {
            "positive": [
                "창의", "혁신", "아이디어", "개선", "효율화", "최적화", "새로운", 
                "도전", "변화", "발상", "창조", "혁신적", "독창적", "창조적"
            ],
            "negative": [
                "보수적", "경직", "틀에박힌", "변화거부", "기존방식", "관습적", 
                "경직된", "고정적", "변화없이"
            ]
        },
        "weight": 0.05,
        "description": "창의성과 혁신 마인드",
        "color": "#FDA7DF"
    },
    "조직적응": {
        "keywords": {
            "positive": [
                "적응", "융화", "조화", "문화", "규칙준수", "윤리", "신뢰", 
                "안정", "일관성", "성실성", "조직", "회사", "팀에"
            ],
            "negative": [
                "부적응", "갈등", "위반", "비윤리", "불신", "일탈", 
                "문제행동", "규정위반", "조직과"
            ]
        },
        "weight": 0.02,
        "description": "조직문화 적응도와 윤리성",
        "color": "#74B9FF"
    }
}

# 개선된 분석 엔진
class AIRISSAnalyzer:
    def __init__(self):
        self.framework = AIRISS_FRAMEWORK
        self.openai_available = False
        self.openai = None
        try:
            import openai
            self.openai = openai
            self.openai_available = True
            logger.info("✅ OpenAI 모듈 로드 성공")
        except ImportError:
            logger.warning("⚠️ OpenAI 모듈 없음 - 키워드 분석만 가능")
    
    def analyze_text(self, text: str, dimension: str) -> Dict[str, Any]:
        """텍스트 분석하여 점수 산출 - 개선된 알고리즘"""
        if not text or text.lower() in ['nan', 'null', '', 'none']:
            return {"score": 50, "confidence": 0, "signals": {"positive": 0, "negative": 0, "positive_words": [], "negative_words": []}}
        
        keywords = self.framework[dimension]["keywords"]
        text_lower = text.lower()
        
        # 키워드 매칭 - 부분 매칭도 포함
        positive_matches = []
        negative_matches = []
        
        for word in keywords["positive"]:
            if word in text_lower:
                positive_matches.append(word)
        
        for word in keywords["negative"]:
            if word in text_lower:
                negative_matches.append(word)
        
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        
        # 점수 계산 (정교한 알고리즘)
        base_score = 50
        positive_boost = min(positive_count * 8, 45)  # 최대 45점 가산
        negative_penalty = min(negative_count * 10, 40)  # 최대 40점 감점
        
        # 텍스트 길이 보정
        text_length = len(text)
        if text_length > 50:
            length_bonus = min((text_length - 50) / 100 * 5, 10)  # 최대 10점 보너스
        else:
            length_bonus = 0
        
        final_score = base_score + positive_boost - negative_penalty + length_bonus
        final_score = max(10, min(100, final_score))  # 10-100 범위로 제한
        
        # 신뢰도 계산
        total_signals = positive_count + negative_count
        base_confidence = min(total_signals * 12, 80)
        length_confidence = min(text_length / 20, 20)  # 텍스트 길이 기반 신뢰도
        confidence = min(base_confidence + length_confidence, 100)
        
        return {
            "score": round(final_score, 1),
            "confidence": round(confidence, 1),
            "signals": {
                "positive": positive_count,
                "negative": negative_count,
                "positive_words": positive_matches[:5],  # 상위 5개
                "negative_words": negative_matches[:5]
            }
        }
    
    def calculate_overall_score(self, dimension_scores: Dict[str, float]) -> Dict[str, Any]:
        """종합 점수 계산 - 개선된 버전"""
        weighted_sum = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            if dimension in self.framework:
                weight = self.framework[dimension]["weight"]
                weighted_sum += score * weight
                total_weight += weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 50
        
        # 등급 산정 - 더 세밀한 구분
        if overall_score >= 95:
            grade = "S급"
            grade_desc = "최우수 (Top 1%)"
            percentile = "상위 1%"
        elif overall_score >= 90:
            grade = "A급"
            grade_desc = "우수 (Top 5%)"
            percentile = "상위 5%"
        elif overall_score >= 85:
            grade = "A-급"
            grade_desc = "우수+ (Top 10%)"
            percentile = "상위 10%"
        elif overall_score >= 80:
            grade = "B급"
            grade_desc = "양호 (Top 20%)"
            percentile = "상위 20%"
        elif overall_score >= 75:
            grade = "B-급"
            grade_desc = "양호- (Top 30%)"
            percentile = "상위 30%"
        elif overall_score >= 70:
            grade = "C급"
            grade_desc = "보통 (Top 40%)"
            percentile = "상위 40%"
        elif overall_score >= 60:
            grade = "D급"
            grade_desc = "개선필요 (Top 60%)"
            percentile = "상위 60%"
        else:
            grade = "F급"
            grade_desc = "집중개선 (하위 40%)"
            percentile = "하위 40%"
        
        return {
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "grade_description": grade_desc,
            "percentile": percentile,
            "weighted_scores": dimension_scores
        }
    
    async def generate_ai_feedback(self, uid: str, opinion: str, api_key: str = None, model: str = "gpt-3.5-turbo", max_tokens: int = 1200) -> Dict[str, Any]:
        """OpenAI를 사용한 상세 AI 피드백 생성 - 완전 개선"""
        logger.info(f"AI 피드백 생성 시작: {uid}, API 키 존재: {bool(api_key)}, 모델: {model}")
        
        if not self.openai_available:
            logger.warning("OpenAI 모듈이 설치되지 않음")
            return {
                "ai_strengths": "OpenAI 모듈이 설치되지 않았습니다. 'pip install openai'로 설치해주세요.",
                "ai_weaknesses": "OpenAI 모듈이 설치되지 않았습니다.",
                "ai_feedback": "키워드 기반 분석만 제공됩니다. OpenAI 모듈을 설치하면 상세한 AI 피드백을 받을 수 있습니다.",
                "processing_time": 0,
                "model_used": "none",
                "tokens_used": 0,
                "error": "OpenAI 모듈 미설치"
            }
        
        if not api_key or api_key.strip() == "":
            logger.warning("OpenAI API 키가 제공되지 않음")
            return {
                "ai_strengths": "OpenAI API 키가 제공되지 않았습니다.",
                "ai_weaknesses": "API 키를 입력하면 상세한 개선점 분석을 받을 수 있습니다.",
                "ai_feedback": "키워드 기반 분석만 수행되었습니다. OpenAI API 키를 입력하면 더 정확하고 상세한 피드백을 받을 수 있습니다.",
                "processing_time": 0,
                "model_used": "none",
                "tokens_used": 0,
                "error": "API 키 없음"
            }
        
        # API 키 유효성 기본 체크
        if not api_key.startswith('sk-'):
            logger.warning("잘못된 API 키 형식")
            return {
                "ai_strengths": "잘못된 API 키 형식입니다.",
                "ai_weaknesses": "올바른 OpenAI API 키를 입력해주세요.",
                "ai_feedback": "API 키는 'sk-'로 시작해야 합니다.",
                "processing_time": 0,
                "model_used": "none",
                "tokens_used": 0,
                "error": "잘못된 API 키 형식"
            }
        
        start_time = datetime.now()
        
        try:
            # OpenAI 클라이언트 초기화
            client = self.openai.OpenAI(api_key=api_key.strip())
            
            # 프롬프트 생성
            prompt = self.create_enhanced_prompt(uid, opinion, model, max_tokens)
            
            logger.info(f"OpenAI API 호출 시작: 모델={model}, 토큰={max_tokens}")
            
            # OpenAI API 호출
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 전문 HR 분석가입니다. AIRISS 8대 영역(업무성과, KPI달성, 태도마인드, 커뮤니케이션, 리더십협업, 전문성학습, 창의혁신, 조직적응)을 기반으로 직원 평가를 분석하고 구체적이고 실행 가능한 피드백을 제공합니다."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=30
            )
            
            feedback_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI API 응답 수신 완료: {len(feedback_text)}자")
            
            # 응답 파싱
            strengths, weaknesses, complete_feedback = self.parse_ai_response(feedback_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "ai_strengths": strengths,
                "ai_weaknesses": weaknesses,
                "ai_feedback": complete_feedback,
                "processing_time": round(processing_time, 2),
                "model_used": model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else max_tokens,
                "error": None
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"OpenAI API 오류: {error_msg}")
            
            # 구체적인 오류 메시지 제공
            if "api_key" in error_msg.lower():
                error_detail = "API 키가 잘못되었거나 만료되었습니다."
            elif "quota" in error_msg.lower():
                error_detail = "API 사용량 한도를 초과했습니다."
            elif "model" in error_msg.lower():
                error_detail = f"모델 '{model}'에 접근할 수 없습니다."
            elif "timeout" in error_msg.lower():
                error_detail = "API 응답 시간이 초과되었습니다."
            else:
                error_detail = f"API 오류: {error_msg}"
            
            return {
                "ai_strengths": f"AI 분석 오류: {error_detail}",
                "ai_weaknesses": "AI 분석을 완료할 수 없습니다.",
                "ai_feedback": f"OpenAI API 호출 중 오류가 발생했습니다: {error_detail}",
                "processing_time": round(processing_time, 2),
                "model_used": model,
                "tokens_used": 0,
                "error": error_detail
            }
    
    def create_enhanced_prompt(self, uid: str, opinion: str, model: str, max_tokens: int) -> str:
        """향상된 AI 프롬프트 생성"""
        # 모델별 최적화
        if "gpt-4" in model:
            return self.create_gpt4_prompt(uid, opinion, max_tokens)
        else:
            return self.create_gpt35_prompt(uid, opinion, max_tokens)
    
    def create_gpt4_prompt(self, uid: str, opinion: str, max_tokens: int) -> str:
        """GPT-4용 상세 프롬프트"""
        return f"""
다음 직원({uid})의 평가 의견을 AIRISS 8대 영역을 기반으로 종합 분석해주세요.

【평가 의견】
{opinion[:2000]}

【AIRISS 8대 영역 (가중치)】
1. 업무성과 (25%) - 업무 산출물의 양과 질
2. KPI달성 (20%) - 핵심성과지표 달성도  
3. 태도마인드 (15%) - 업무에 대한 태도와 마인드셋
4. 커뮤니케이션 (15%) - 의사소통 능력과 스타일
5. 리더십협업 (10%) - 리더십과 협업 능력
6. 전문성학습 (8%) - 전문성과 학습능력
7. 창의혁신 (5%) - 창의성과 혁신 마인드
8. 조직적응 (2%) - 조직문화 적응도와 윤리성

【출력 형식】
[장점]
1. 구체적 장점 1 (관련 AIRISS 영역 명시)
2. 구체적 장점 2 (관련 AIRISS 영역 명시)
3. 구체적 장점 3 (관련 AIRISS 영역 명시)

[개선점]
1. 구체적 개선점 1 (관련 AIRISS 영역 명시)
2. 구체적 개선점 2 (관련 AIRISS 영역 명시)
3. 구체적 개선점 3 (관련 AIRISS 영역 명시)

[종합 피드백]
AIRISS 8대 영역을 종합하여 실행 가능한 피드백을 600-800자로 작성:
- 핵심 강점과 활용 방안
- 우선 개선 영역과 구체적 방법
- 향후 6개월 발전 계획
- 조직 기여도 향상 방안

반드시 각 섹션을 완전히 작성하고 실무에 바로 적용할 수 있도록 구체적으로 작성해주세요.
        """
    
    def create_gpt35_prompt(self, uid: str, opinion: str, max_tokens: int) -> str:
        """GPT-3.5용 효율적 프롬프트"""
        return f"""
직원 {uid} 평가분석:

의견: {opinion[:1200]}

AIRISS 8영역으로 분석:
1.업무성과(25%) 2.KPI달성(20%) 3.태도마인드(15%) 4.커뮤니케이션(15%) 
5.리더십협업(10%) 6.전문성학습(8%) 7.창의혁신(5%) 8.조직적응(2%)

형식:
[장점]
1. 핵심장점1 (영역명시)
2. 핵심장점2 (영역명시)
3. 핵심장점3 (영역명시)

[개선점]
1. 개선점1 (영역명시)
2. 개선점2 (영역명시)
3. 개선점3 (영역명시)

[종합 피드백]
실행가능한 구체적 피드백 400-600자:
- 강점 활용법
- 개선 방안
- 성장 계획

완전한 문장으로 끝내주세요.
        """
    
    def parse_ai_response(self, response: str) -> tuple:
        """AI 응답 파싱 - 개선된 버전"""
        try:
            # 섹션별로 분할
            sections = response.split('[')
            
            strengths = ""
            weaknesses = ""
            feedback = ""
            
            for section in sections:
                section = section.strip()
                if section.startswith('장점]'):
                    content = section.replace('장점]', '').strip()
                    # 다음 섹션 시작 전까지만 추출
                    if '[' in content:
                        content = content.split('[')[0].strip()
                    strengths = content
                        
                elif section.startswith('개선점]'):
                    content = section.replace('개선점]', '').strip()
                    if '[' in content:
                        content = content.split('[')[0].strip()
                    weaknesses = content
                        
                elif section.startswith('종합 피드백]') or section.startswith('종합피드백]'):
                    content = section.replace('종합 피드백]', '').replace('종합피드백]', '').strip()
                    feedback = content
            
            # 빈 값 처리 및 기본값 설정
            if not strengths.strip():
                strengths = "텍스트 분석을 통해 다음과 같은 긍정적 특성이 관찰됩니다."
            if not weaknesses.strip():
                weaknesses = "전반적으로 우수하나, 지속적인 성장을 위한 개선 기회가 있습니다."
            if not feedback.strip():
                feedback = response  # 전체 응답 사용
            
            # 텍스트 정리
            strengths = self.clean_text(strengths)
            weaknesses = self.clean_text(weaknesses)
            feedback = self.clean_text(feedback)
                
            return strengths, weaknesses, feedback
            
        except Exception as e:
            logger.error(f"AI 응답 파싱 오류: {e}")
            return "장점 분석 중 오류 발생", "개선점 분석 중 오류 발생", response
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 여러 줄바꿈을 하나로 통합
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # 너무 긴 텍스트 자르기
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text.strip()

analyzer = AIRISSAnalyzer()

# 데이터 모델 - 개선된 버전
class AnalysisRequest(BaseModel):
    file_id: str
    sample_size: int = 10
    analysis_mode: str = "standard"
    openai_api_key: Optional[str] = None
    enable_ai_feedback: bool = False
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1200

# 메인 HTML 템플릿 - AI 피드백 기능 개선
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """메인 페이지 - AI 피드백 기능 개선"""
    html_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIRISS | AI-Powered Employee Analytics</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .header {
            text-align: center; color: white; margin-bottom: 40px; padding: 40px 20px;
        }
        
        .header h1 {
            font-size: 3.5rem; font-weight: 700; margin-bottom: 10px;
            text-shadow: 0 2px 20px rgba(0,0,0,0.3);
        }
        
        .header .subtitle { font-size: 1.3rem; opacity: 0.9; margin-bottom: 30px; }
        
        .status-badge {
            display: inline-block; background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px); padding: 12px 24px; border-radius: 50px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .main-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;
        }
        
        @media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; } }
        
        .card {
            background: rgba(255,255,255,0.95); backdrop-filter: blur(20px);
            border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2); transition: all 0.3s ease;
        }
        
        .card:hover { transform: translateY(-5px); box-shadow: 0 30px 80px rgba(0,0,0,0.15); }
        
        .card h3 {
            color: #2d3748; font-size: 1.5rem; margin-bottom: 20px;
            display: flex; align-items: center; gap: 10px;
        }
        
        .card h3 i {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .dimensions-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-top: 20px;
        }
        
        .dimension-card {
            background: linear-gradient(135deg, #f8f9ff, #fff); border: 2px solid #e2e8f0;
            border-radius: 12px; padding: 15px; text-align: center; transition: all 0.3s ease;
            position: relative; overflow: hidden;
        }
        
        .dimension-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
            background: var(--color);
        }
        
        .dimension-card h4 {
            font-size: 0.9rem; font-weight: 600; margin-bottom: 5px; color: #2d3748;
        }
        
        .dimension-card p { font-size: 0.8rem; color: #718096; line-height: 1.4; }
        
        .dimension-card .weight {
            position: absolute; top: 8px; right: 8px; background: var(--color);
            color: white; font-size: 0.7rem; padding: 2px 6px; border-radius: 10px;
            font-weight: bold;
        }
        
        .upload-area {
            border: 3px dashed #cbd5e0; border-radius: 16px; padding: 40px 20px;
            text-align: center; background: linear-gradient(135deg, #f7fafc, #edf2f7);
            transition: all 0.3s ease; cursor: pointer; position: relative; overflow: hidden;
        }
        
        .upload-area:hover {
            border-color: #667eea; background: linear-gradient(135deg, #e6fffa, #f0fff4);
        }
        
        .upload-area.dragover {
            border-color: #48bb78; background: linear-gradient(135deg, #f0fff4, #c6f6d5);
            transform: scale(1.02);
        }
        
        .upload-area input[type="file"] {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            opacity: 0; cursor: pointer;
        }
        
        .upload-content { pointer-events: none; }
        .upload-content i { font-size: 3rem; color: #667eea; margin-bottom: 15px; }
        
        .btn {
            background: linear-gradient(135deg, #667eea, #764ba2); color: white;
            border: none; padding: 14px 28px; border-radius: 12px; font-size: 1rem;
            font-weight: 600; cursor: pointer; transition: all 0.3s ease;
            text-decoration: none; display: inline-block; position: relative;
            overflow: hidden;
        }
        
        .btn:hover {
            transform: translateY(-2px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active { transform: translateY(0); }
        
        .btn:disabled {
            background: #a0aec0; cursor: not-allowed; transform: none; box-shadow: none;
        }
        
        .btn-success { background: linear-gradient(135deg, #48bb78, #38a169); }
        .btn-success:hover { box-shadow: 0 10px 30px rgba(72, 187, 120, 0.4); }
        
        .result-card {
            background: linear-gradient(135deg, #d4edda, #c3e6cb); border: 1px solid #c3e6cb;
            border-radius: 12px; padding: 20px; margin: 20px 0; animation: slideInUp 0.5s ease;
        }
        
        .error-card {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 1px solid #f5c6cb;
            border-radius: 12px; padding: 20px; margin: 20px 0; animation: slideInUp 0.5s ease;
        }
        
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.9); border-radius: 12px; padding: 20px;
            text-align: center; border: 2px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px); border-color: rgba(102, 126, 234, 0.3);
        }
        
        .stat-number {
            font-size: 2.2rem; font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 5px;
        }
        
        .stat-label { color: #718096; font-size: 0.9rem; font-weight: 500; }
        
        .progress-container { margin: 20px 0; display: none; }
        
        .progress-bar {
            background: #e2e8f0; border-radius: 10px; height: 8px;
            overflow: hidden; position: relative;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2); height: 100%;
            width: 0%; transition: width 0.3s ease; position: relative;
        }
        
        .progress-fill::after {
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        .analysis-controls {
            display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;
        }
        
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { font-weight: 600; color: #2d3748; }
        
        select, input[type="number"], input[type="password"] {
            padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px;
            font-size: 1rem; transition: border-color 0.3s ease;
        }
        
        select:focus, input[type="number"]:focus, input[type="password"]:focus {
            outline: none; border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .log-container {
            background: #1a202c; color: #e2e8f0; border-radius: 12px; padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9rem;
            max-height: 300px; overflow-y: auto; margin: 20px 0; display: none;
        }
        
        .log-entry {
            margin-bottom: 5px; opacity: 0; animation: fadeInLog 0.3s ease forwards;
        }
        
        .log-timestamp { color: #68d391; margin-right: 10px; }
        .full-width { grid-column: 1 / -1; }
        
        /* AI 설정 스타일 개선 */
        .ai-settings {
            background: linear-gradient(135deg, #e3f2fd, #f1f8e9); padding: 25px;
            border-radius: 15px; margin: 25px 0; border: 2px solid rgba(102, 126, 234, 0.2);
        }
        
        .ai-toggle {
            display: flex; align-items: center; gap: 15px; cursor: pointer;
            padding: 15px; border-radius: 10px; background: rgba(255,255,255,0.7);
            border: 2px solid transparent; transition: all 0.3s ease;
        }
        
        .ai-toggle:hover {
            background: rgba(255,255,255,0.9); border-color: rgba(102, 126, 234, 0.3);
        }
        
        .ai-toggle input[type="checkbox"] {
            width: 20px; height: 20px; accent-color: #667eea;
        }
        
        .ai-status {
            display: inline-block; padding: 8px 16px; border-radius: 20px;
            font-size: 0.9rem; font-weight: 600; margin-left: auto;
        }
        
        .ai-status.enabled { background: #d4edda; color: #155724; }
        .ai-status.disabled { background: #f8d7da; color: #721c24; }
        
        .ai-advanced-settings {
            margin-top: 20px; padding: 20px; background: rgba(255,255,255,0.5);
            border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        .cost-estimate {
            background: #fff3cd; border: 1px solid #ffeeba; border-radius: 8px;
            padding: 15px; margin: 15px 0; font-size: 0.9rem;
        }
        
        .cost-estimate strong { color: #856404; }
        
        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes fadeInLog { to { opacity: 1; } }
        
        .download-section {
            text-align: center; padding: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            border-radius: 16px; margin-top: 20px;
        }
        
        .features-list { list-style: none; margin: 15px 0; }
        .features-list li { padding: 5px 0; color: #718096; font-size: 0.9rem; }
        .features-list li::before { content: '✅'; margin-right: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> AIRISS</h1>
            <div class="subtitle">AI-Powered Employee Performance & Competency Analysis</div>
            <div class="status-badge">
                <i class="fas fa-check-circle"></i> Enterprise Ready • Version 3.1 • AI 피드백 개선판
            </div>
        </div>
        
        <div class="main-grid">
            <!-- 분석 영역 카드 -->
            <div class="card">
                <h3><i class="fas fa-chart-line"></i> 8대 분석 영역</h3>
                <div class="dimensions-grid">
                    <div class="dimension-card" style="--color: #FF6B6B;">
                        <div class="weight">25%</div>
                        <h4>📊 업무성과</h4>
                        <p>업무 산출물의 양과 질</p>
                    </div>
                    <div class="dimension-card" style="--color: #4ECDC4;">
                        <div class="weight">20%</div>
                        <h4>🎯 KPI달성</h4>
                        <p>핵심성과지표 달성도</p>
                    </div>
                    <div class="dimension-card" style="--color: #45B7D1;">
                        <div class="weight">15%</div>
                        <h4>🧠 태도마인드</h4>
                        <p>업무 태도와 마인드셋</p>
                    </div>
                    <div class="dimension-card" style="--color: #96CEB4;">
                        <div class="weight">15%</div>
                        <h4>💬 커뮤니케이션</h4>
                        <p>의사소통 능력</p>
                    </div>
                    <div class="dimension-card" style="--color: #FFEAA7;">
                        <div class="weight">10%</div>
                        <h4>👥 리더십협업</h4>
                        <p>리더십과 협업 능력</p>
                    </div>
                    <div class="dimension-card" style="--color: #DDA0DD;">
                        <div class="weight">8%</div>
                        <h4>📚 전문성학습</h4>
                        <p>전문성과 학습능력</p>
                    </div>
                    <div class="dimension-card" style="--color: #FDA7DF;">
                        <div class="weight">5%</div>
                        <h4>💡 창의혁신</h4>
                        <p>창의성과 혁신 마인드</p>
                    </div>
                    <div class="dimension-card" style="--color: #74B9FF;">
                        <div class="weight">2%</div>
                        <h4>🏢 조직적응</h4>
                        <p>조직문화 적응도</p>
                    </div>
                </div>
            </div>
            
            <!-- 파일 업로드 카드 -->
            <div class="card">
                <h3><i class="fas fa-cloud-upload-alt"></i> 데이터 업로드</h3>
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" multiple>
                    <div class="upload-content">
                        <i class="fas fa-file-upload"></i>
                        <h4>파일을 드래그하거나 클릭하여 선택하세요</h4>
                        <p>Excel (.xlsx, .xls) 또는 CSV 파일을 지원합니다</p>
                    </div>
                </div>
                <div id="uploadResult"></div>
                <button class="btn" onclick="processUpload()" id="uploadBtn" disabled>
                    <i class="fas fa-magic"></i> 데이터 분석 시작
                </button>
            </div>
        </div>
        
        <!-- 분석 설정 -->
        <div class="card full-width" id="analysisCard" style="display: none;">
            <h3><i class="fas fa-cogs"></i> 분석 설정</h3>
            <div class="analysis-controls">
                <div class="form-group">
                    <label for="sampleSize">분석 샘플 수</label>
                    <select id="sampleSize">
                        <option value="10">10개 (빠른 테스트)</option>
                        <option value="25" selected>25개 (표준)</option>
                        <option value="50">50개 (상세)</option>
                        <option value="100">100개 (정밀)</option>
                        <option value="all">전체 데이터</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="analysisMode">분석 모드</label>
                    <select id="analysisMode">
                        <option value="basic">기본 분석 (키워드만)</option>
                        <option value="standard" selected>표준 분석 (키워드 + AI)</option>
                        <option value="comprehensive">종합 분석 (완전한 AI)</option>
                    </select>
                </div>
            </div>
            
            <!-- AI 설정 섹션 - 개선된 UI -->
            <div class="ai-settings">
                <h4 style="margin-bottom: 20px; color: #2d3748;"><i class="fas fa-robot"></i> AI 피드백 설정</h4>
                
                <div class="ai-toggle" onclick="toggleAISettings()">
                    <input type="checkbox" id="enableAI" onclick="event.stopPropagation();">
                    <div style="flex: 1;">
                        <strong>OpenAI GPT를 사용한 상세 AI 피드백</strong>
                        <p style="margin: 5px 0 0 0; color: #718096; font-size: 0.9rem;">
                            더 정확하고 구체적인 개인별 피드백을 받을 수 있습니다
                        </p>
                    </div>
                    <div class="ai-status disabled" id="aiStatus">비활성화</div>
                </div>
                
                <div id="aiAdvancedSettings" class="ai-advanced-settings" style="display: none;">
                    <div class="analysis-controls">
                        <div class="form-group">
                            <label for="openaiModel">AI 모델 선택</label>
                            <select id="openaiModel" onchange="updateCostEstimate()">
                                <option value="gpt-3.5-turbo" selected>GPT-3.5 Turbo (빠름, 저비용)</option>
                                <option value="gpt-4-turbo">GPT-4 Turbo (느림, 고품질)</option>
                                <option value="gpt-4">GPT-4 (균형)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="maxTokens">응답 길이</label>
                            <select id="maxTokens" onchange="updateCostEstimate()">
                                <option value="800">간단 (800토큰)</option>
                                <option value="1200" selected>표준 (1200토큰)</option>
                                <option value="1500">상세 (1500토큰)</option>
                                <option value="2000">완전 (2000토큰)</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group" style="margin-top: 20px;">
                        <label for="openaiKey">
                            <i class="fas fa-key"></i> OpenAI API 키
                            <span style="color: #e53e3e;">*</span>
                        </label>
                        <input type="password" id="openaiKey" placeholder="sk-proj-..." 
                               style="width: 100%;" onchange="validateApiKey()">
                        <small style="color: #718096; margin-top: 8px; display: block;">
                            <i class="fas fa-shield-alt"></i> API 키는 안전하게 처리되며 서버에 저장되지 않습니다
                        </small>
                    </div>
                    
                    <div class="cost-estimate" id="costEstimate">
                        <strong>💰 예상 비용:</strong> 25개 샘플 기준 약 $0.10 (GPT-3.5 Turbo, 1200토큰)
                    </div>
                    
                    <div style="margin-top: 15px; font-size: 0.95rem; color: #4a5568; line-height: 1.6;">
                        <strong>📊 모델별 특징:</strong><br>
                        • <strong>GPT-3.5 Turbo:</strong> 빠른 속도, 합리적 품질, 낮은 비용<br>
                        • <strong>GPT-4:</strong> 높은 품질, 균형잡힌 성능, 중간 비용<br>
                        • <strong>GPT-4 Turbo:</strong> 최고 품질, 깊이 있는 분석, 높은 비용
                    </div>
                </div>
            </div>
            
            <button class="btn btn-success" onclick="startAnalysis()" id="analyzeBtn">
                <i class="fas fa-rocket"></i> AIRISS 분석 실행
            </button>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">분석 준비 중...</p>
            </div>
            
            <div class="log-container" id="logContainer"></div>
            
            <div id="analysisResult"></div>
        </div>
        
        <!-- 결과 다운로드 -->
        <div class="card full-width download-section" id="downloadCard" style="display: none;">
            <h3><i class="fas fa-download"></i> 분석 결과 다운로드</h3>
            <div id="downloadContent"></div>
        </div>
    </div>

    <script>
        // 전역 변수
        let currentFileData = null;
        let analysisJobId = null;
        
        // 유틸리티 함수
        function formatNumber(num) {
            return new Intl.NumberFormat('ko-KR').format(num);
        }
        
        function getCurrentTime() {
            return new Date().toLocaleTimeString('ko-KR');
        }
        
        function addLog(message) {
            const logContainer = document.getElementById('logContainer');
            logContainer.style.display = 'block';
            
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `<span class="log-timestamp">[${getCurrentTime()}]</span> ${message}`;
            
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        function updateProgress(percentage, text) {
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressContainer.style.display = 'block';
            progressFill.style.width = percentage + '%';
            progressText.textContent = text;
        }
        
        // AI 설정 관련 함수들
        function toggleAISettings() {
            const checkbox = document.getElementById('enableAI');
            const settings = document.getElementById('aiAdvancedSettings');
            const status = document.getElementById('aiStatus');
            
            checkbox.checked = !checkbox.checked;
            
            if (checkbox.checked) {
                settings.style.display = 'block';
                status.textContent = '활성화';
                status.className = 'ai-status enabled';
                addLog('🤖 AI 피드백 기능 활성화됨');
                updateCostEstimate();
            } else {
                settings.style.display = 'none';
                status.textContent = '비활성화';
                status.className = 'ai-status disabled';
                addLog('📊 키워드 분석만 사용됨');
            }
        }
        
        function validateApiKey() {
            const apiKey = document.getElementById('openaiKey').value.trim();
            const input = document.getElementById('openaiKey');
            
            if (apiKey === '') {
                input.style.borderColor = '#e2e8f0';
                return false;
            } else if (!apiKey.startsWith('sk-')) {
                input.style.borderColor = '#e53e3e';
                addLog('⚠️ API 키는 "sk-"로 시작해야 합니다');
                return false;
            } else {
                input.style.borderColor = '#48bb78';
                addLog('✅ API 키 형식이 올바릅니다');
                return true;
            }
        }
        
        function updateCostEstimate() {
            const model = document.getElementById('openaiModel').value;
            const tokens = parseInt(document.getElementById('maxTokens').value);
            const sampleSize = document.getElementById('sampleSize').value;
            const samples = sampleSize === 'all' ? 100 : parseInt(sampleSize); // 예상값
            
            let costPerToken = 0;
            if (model === 'gpt-3.5-turbo') {
                costPerToken = 0.002 / 1000; // $0.002 per 1K tokens
            } else if (model === 'gpt-4-turbo') {
                costPerToken = 0.01 / 1000; // $0.01 per 1K tokens
            } else if (model === 'gpt-4') {
                costPerToken = 0.03 / 1000; // $0.03 per 1K tokens
            }
            
            const estimatedCost = samples * tokens * costPerToken;
            const costText = estimatedCost < 0.01 ? '< $0.01' : `$${estimatedCost.toFixed(2)}`;
            
            document.getElementById('costEstimate').innerHTML = `
                <strong>💰 예상 비용:</strong> ${samples}개 샘플 기준 약 ${costText} (${model}, ${tokens}토큰)
                <br><small>실제 비용은 응답 길이에 따라 달라질 수 있습니다</small>
            `;
        }
        
        // 파일 업로드 관련
        function setupFileUpload() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            // 드래그 앤 드롭
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    handleFileSelection();
                }
            });
            
            // 파일 선택
            fileInput.addEventListener('change', handleFileSelection);
        }
        
        function handleFileSelection() {
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');
            const file = fileInput.files[0];
            
            if (file) {
                document.getElementById('uploadResult').innerHTML = `
                    <div class="result-card">
                        <h4><i class="fas fa-file-check"></i> 파일 선택됨</h4>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-number">${file.name}</div>
                                <div class="stat-label">파일명</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number">${(file.size / 1024 / 1024).toFixed(2)}MB</div>
                                <div class="stat-label">파일 크기</div>
                            </div>
                        </div>
                    </div>
                `;
                uploadBtn.disabled = false;
                addLog(`파일 선택: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
            }
        }
        
        async function processUpload() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('파일을 선택해주세요.');
                return;
            }
            
            const uploadBtn = document.getElementById('uploadBtn');
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 업로드 중...';
            
            addLog('서버로 파일 업로드 시작...');
            updateProgress(20, '파일 업로드 중...');
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    currentFileData = result;
                    addLog(`✅ 업로드 성공: ${result.total_records}개 레코드 감지`);
                    updateProgress(100, '업로드 완료!');
                    
                    displayUploadResult(result);
                    showAnalysisCard();
                } else {
                    throw new Error(result.detail || '업로드 실패');
                }
            } catch (error) {
                addLog(`❌ 업로드 오류: ${error.message}`);
                document.getElementById('uploadResult').innerHTML = `
                    <div class="error-card">
                        <h4><i class="fas fa-exclamation-triangle"></i> 업로드 실패</h4>
                        <p>${error.message}</p>
                    </div>
                `;
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<i class="fas fa-magic"></i> 데이터 분석 시작';
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                }, 2000);
            }
        }
        
        function displayUploadResult(data) {
            document.getElementById('uploadResult').innerHTML = `
                <div class="result-card">
                    <h4><i class="fas fa-check-circle"></i> 데이터 검증 완료</h4>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">${formatNumber(data.total_records)}</div>
                            <div class="stat-label">총 레코드</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.column_count}</div>
                            <div class="stat-label">컬럼 수</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.uid_columns.length}</div>
                            <div class="stat-label">ID 컬럼</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${data.opinion_columns.length}</div>
                            <div class="stat-label">의견 컬럼</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>AIRISS 분석 준비:</strong> 
                        ${data.airiss_ready ? 
                            '<span style="color: #48bb78;"><i class="fas fa-check"></i> 완전 준비됨</span>' : 
                            '<span style="color: #e53e3e;"><i class="fas fa-times"></i> 컬럼 확인 필요</span>'
                        }
                    </div>
                </div>
            `;
        }
        
        function showAnalysisCard() {
            document.getElementById('analysisCard').style.display = 'block';
            document.getElementById('analysisCard').scrollIntoView({ behavior: 'smooth' });
        }
        
        async function startAnalysis() {
            if (!currentFileData) {
                alert('먼저 파일을 업로드해주세요.');
                return;
            }
            
            const sampleSize = document.getElementById('sampleSize').value;
            const analysisMode = document.getElementById('analysisMode').value;
            const enableAI = document.getElementById('enableAI').checked;
            const openaiKey = document.getElementById('openaiKey').value.trim();
            const openaiModel = document.getElementById('openaiModel').value;
            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            // AI 피드백 설정 검증
            if (enableAI && !openaiKey) {
                alert('AI 피드백을 사용하려면 OpenAI API 키를 입력해주세요.');
                document.getElementById('openaiKey').focus();
                return;
            }
            
            if (enableAI && !validateApiKey()) {
                alert('올바른 OpenAI API 키를 입력해주세요.');
                document.getElementById('openaiKey').focus();
                return;
            }
            
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 분석 진행 중...';
            
            addLog('🚀 AIRISS AI 분석 시작...');
            addLog(`📊 설정: ${sampleSize}개 샘플, ${analysisMode} 모드`);
            if (enableAI) {
                addLog(`🤖 AI 피드백: ${openaiModel} 모델, ${maxTokens} 토큰`);
            } else {
                addLog('📈 키워드 기반 분석만 수행');
            }
            updateProgress(0, 'AI 분석 엔진 초기화...');
            
            try {
                const requestData = {
                    file_id: currentFileData.file_id,
                    sample_size: sampleSize === 'all' ? currentFileData.total_records : parseInt(sampleSize),
                    analysis_mode: analysisMode,
                    enable_ai_feedback: enableAI,
                    openai_api_key: enableAI ? openaiKey : null,
                    openai_model: openaiModel,
                    max_tokens: maxTokens
                };
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    analysisJobId = result.job_id;
                    addLog(`✅ 분석 작업 시작: ${result.job_id}`);
                    
                    // 진행상황 폴링
                    pollAnalysisProgress(result.job_id);
                } else {
                    throw new Error(result.detail || '분석 시작 실패');
                }
            } catch (error) {
                addLog(`❌ 분석 오류: ${error.message}`);
                document.getElementById('analysisResult').innerHTML = `
                    <div class="error-card">
                        <h4><i class="fas fa-exclamation-triangle"></i> 분석 실패</h4>
                        <p>${error.message}</p>
                    </div>
                `;
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> AIRISS 분석 실행';
            }
        }
        
        async function pollAnalysisProgress(jobId) {
            const pollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/status/${jobId}`);
                    const status = await response.json();
                    
                    const progress = status.progress || 0;
                    updateProgress(progress, `분석 진행: ${status.processed}/${status.total} (${progress.toFixed(1)}%)`);
                    
                    if (status.status === 'completed') {
                        clearInterval(pollInterval);
                        addLog('🎉 AIRISS 분석 완료!');
                        displayAnalysisResult(status);
                        showDownloadCard(jobId);
                        
                        const analyzeBtn = document.getElementById('analyzeBtn');
                        analyzeBtn.disabled = false;
                        analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> AIRISS 분석 실행';
                        
                    } else if (status.status === 'failed') {
                        clearInterval(pollInterval);
                        addLog(`❌ 분석 실패: ${status.error}`);
                        
                        const analyzeBtn = document.getElementById('analyzeBtn');
                        analyzeBtn.disabled = false;
                        analyzeBtn.innerHTML = '<i class="fas fa-rocket"></i> AIRISS 분석 실행';
                        
                    } else if (status.status === 'processing') {
                        addLog(`⏳ 진행 중: ${status.processed}/${status.total} 레코드 분석`);
                    }
                    
                } catch (error) {
                    addLog(`⚠️ 상태 확인 오류: ${error.message}`);
                }
            }, 2000);
        }
        
        function displayAnalysisResult(status) {
            document.getElementById('analysisResult').innerHTML = `
                <div class="result-card">
                    <h4><i class="fas fa-chart-bar"></i> 분석 결과 요약</h4>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">${formatNumber(status.processed)}</div>
                            <div class="stat-label">성공 분석</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${status.failed || 0}</div>
                            <div class="stat-label">실패 분석</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${((status.processed / status.total) * 100).toFixed(1)}%</div>
                            <div class="stat-label">성공률</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${status.average_score || 0}</div>
                            <div class="stat-label">평균 점수</div>
                        </div>
                    </div>
                    <p style="text-align: center; margin-top: 20px; color: #718096;">
                        <i class="fas fa-clock"></i> 처리 시간: ${status.processing_time || '계산 중...'}
                    </p>
                </div>
            `;
        }
        
        function showDownloadCard(jobId) {
            const downloadCard = document.getElementById('downloadCard');
            downloadCard.style.display = 'block';
            
            document.getElementById('downloadContent').innerHTML = `
                <button class="btn btn-success" onclick="downloadResult('${jobId}')" style="font-size: 1.2rem; padding: 18px 36px;">
                    <i class="fas fa-download"></i> 완전한 AIRISS 분석 결과 다운로드
                </button>
                <ul class="features-list" style="margin-top: 20px; text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
                    <li>8대 영역별 정량 점수</li>
                    <li>종합 등급 및 백분위</li>
                    <li>키워드 기반 상세 분석</li>
                    <li>AI 피드백 및 개선방향</li>
                    <li>통계 요약 정보</li>
                </ul>
            `;
            
            downloadCard.scrollIntoView({ behavior: 'smooth' });
        }
        
        function downloadResult(jobId) {
            addLog('📥 분석 결과 다운로드 시작...');
            window.open(`/download/${jobId}`, '_blank');
        }
        
        // 초기화
        document.addEventListener('DOMContentLoaded', function() {
            setupFileUpload();
            addLog('🎯 AIRISS 시스템 초기화 완료');
            addLog('📁 파일을 업로드하여 분석을 시작하세요');
            addLog('🤖 AI 피드백을 원하시면 설정에서 활성화해주세요');
        });
    </script>
</body>
</html>
    """
    return html_content

# API 엔드포인트들 - 개선된 버전
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 기초 분석"""
    try:
        logger.info(f"파일 업로드 시작: {file.filename}")
        
        # 파일 내용 읽기
        contents = await file.read()
        
        # 파일 타입에 따른 처리
        if file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
            logger.info("Excel 파일 처리 완료")
        elif file.filename.endswith('.csv'):
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                    logger.info(f"CSV 파일 처리 완료 (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"인코딩 {encoding} 실패: {e}")
                    continue
            
            if df is None:
                raise HTTPException(status_code=400, detail="CSV 파일 인코딩을 인식할 수 없습니다")
        else:
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다")
        
        # 파일 ID 생성 및 저장
        file_id = str(uuid.uuid4())
        
        # 필요한 폴더 생성
        os.makedirs('temp', exist_ok=True)
        
        # 컬럼 분석
        all_columns = list(df.columns)
        uid_columns = [col for col in all_columns if any(keyword in col.lower() 
                      for keyword in ['uid', 'id', '아이디', '사번', '직원', 'user', 'emp'])]
        opinion_columns = [col for col in all_columns if any(keyword in col.lower() 
                          for keyword in ['의견', 'opinion', '평가', 'feedback', '내용', '코멘트', '피드백', 'comment', 'review'])]
        
        # 데이터 품질 체크
        total_records = len(df)
        non_empty_records = len(df.dropna(subset=opinion_columns if opinion_columns else []))
        
        # 저장
        store.add_file(file_id, {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': datetime.now(),
            'total_records': total_records,
            'columns': all_columns,
            'uid_columns': uid_columns,
            'opinion_columns': opinion_columns
        })
        
        logger.info(f"파일 저장 완료: {file_id}")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "total_records": total_records,
            "column_count": len(all_columns),
            "uid_columns": uid_columns,
            "opinion_columns": opinion_columns,
            "airiss_ready": len(uid_columns) > 0 and len(opinion_columns) > 0,
            "data_quality": {
                "non_empty_records": non_empty_records,
                "completeness": round((non_empty_records / total_records) * 100, 1) if total_records > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"파일 업로드 오류: {e}")
        raise HTTPException(status_code=400, detail=f"파일 처리 오류: {str(e)}")

@app.post("/analyze")
async def start_analysis(request: AnalysisRequest):
    """분석 작업 시작 - AI 피드백 기능 개선"""
    try:
        # 파일 데이터 확인
        file_data = store.get_file(request.file_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        
        # 작업 ID 생성
        job_id = str(uuid.uuid4())
        
        # AI 피드백 설정 로깅
        logger.info(f"분석 요청: AI 피드백={request.enable_ai_feedback}, API 키 존재={bool(request.openai_api_key)}")
        
        # 작업 정보 초기화
        store.add_job(job_id, {
            "status": "processing",
            "file_id": request.file_id,
            "sample_size": request.sample_size,
            "analysis_mode": request.analysis_mode,
            "enable_ai_feedback": request.enable_ai_feedback,
            "openai_api_key": request.openai_api_key,
            "openai_model": request.openai_model,
            "max_tokens": request.max_tokens,
            "start_time": datetime.now(),
            "total": request.sample_size,
            "processed": 0,
            "failed": 0,
            "progress": 0.0,
            "results": []
        })
        
        # 백그라운드에서 분석 실행
        asyncio.create_task(process_analysis(job_id))
        
        logger.info(f"분석 작업 시작: {job_id}")
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "AIRISS 분석이 시작되었습니다",
            "ai_feedback_enabled": request.enable_ai_feedback
        }
        
    except Exception as e:
        logger.error(f"분석 시작 오류: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def process_analysis(job_id: str):
    """백그라운드 분석 처리 - AI 피드백 완전 개선"""
    try:
        job_data = store.get_job(job_id)
        file_data = store.get_file(job_data["file_id"])
        
        df = file_data["dataframe"]
        sample_size = job_data["sample_size"]
        enable_ai = job_data.get("enable_ai_feedback", False)
        api_key = job_data.get("openai_api_key", None)
        model = job_data.get("openai_model", "gpt-3.5-turbo")
        max_tokens = job_data.get("max_tokens", 1200)
        
        logger.info(f"분석 처리 시작: 샘플={sample_size}, AI={enable_ai}, 모델={model}")
        
        # 샘플 데이터 선택
        if sample_size == "all" or sample_size >= len(df):
            sample_df = df.copy()
        else:
            sample_df = df.head(sample_size).copy()
        
        # 컬럼 확인
        uid_cols = file_data["uid_columns"]
        opinion_cols = file_data["opinion_columns"]
        
        if not uid_cols or not opinion_cols:
            store.update_job(job_id, {
                "status": "failed",
                "error": "필수 컬럼(UID, 의견)을 찾을 수 없습니다"
            })
            return
        
        results = []
        total_rows = len(sample_df)
        ai_success_count = 0
        ai_fail_count = 0
        
        logger.info(f"분석 시작: {total_rows}개 레코드, AI 피드백: {enable_ai}")
        
        for idx, row in sample_df.iterrows():
            try:
                # UID와 의견 추출
                uid = str(row[uid_cols[0]]) if uid_cols else f"user_{idx}"
                opinion = str(row[opinion_cols[0]]) if opinion_cols else ""
                
                # 빈 의견 처리
                if not opinion or opinion.lower() in ['nan', 'null', '', 'none']:
                    store.update_job(job_id, {"failed": job_data["failed"] + 1})
                    continue
                
                # 8대 영역별 키워드 분석
                dimension_scores = {}
                dimension_details = {}
                
                for dimension in AIRISS_FRAMEWORK.keys():
                    analysis_result = analyzer.analyze_text(opinion, dimension)
                    dimension_scores[dimension] = analysis_result["score"]
                    dimension_details[dimension] = analysis_result
                
                # 종합 점수 계산
                overall_analysis = analyzer.calculate_overall_score(dimension_scores)
                
                # 결과 레코드 생성
                result_record = {
                    "UID": uid,
                    "원본의견": opinion[:500] + "..." if len(opinion) > 500 else opinion,
                    "AIRISS_종합점수": overall_analysis["overall_score"],
                    "AIRISS_등급": overall_analysis["grade"],
                    "등급설명": overall_analysis["grade_description"],
                    "백분위": overall_analysis["percentile"]
                }
                
                # 8대 영역별 점수 추가
                for dimension, score in dimension_scores.items():
                    result_record[f"{dimension}_점수"] = score
                    details = dimension_details[dimension]
                    result_record[f"{dimension}_신뢰도"] = details["confidence"]
                    result_record[f"{dimension}_긍정신호"] = details["signals"]["positive"]
                    result_record[f"{dimension}_부정신호"] = details["signals"]["negative"]
                    # 키워드도 추가
                    if details["signals"]["positive_words"]:
                        result_record[f"{dimension}_긍정키워드"] = ", ".join(details["signals"]["positive_words"])
                    if details["signals"]["negative_words"]:
                        result_record[f"{dimension}_부정키워드"] = ", ".join(details["signals"]["negative_words"])
                
                # AI 피드백 생성 (활성화된 경우)
                if enable_ai and api_key:
                    logger.info(f"AI 피드백 생성: {uid}")
                    ai_feedback = await analyzer.generate_ai_feedback(uid, opinion, api_key, model, max_tokens)
                    
                    result_record["AI_장점"] = ai_feedback["ai_strengths"]
                    result_record["AI_개선점"] = ai_feedback["ai_weaknesses"]
                    result_record["AI_종합피드백"] = ai_feedback["ai_feedback"]
                    result_record["AI_처리시간"] = ai_feedback["processing_time"]
                    result_record["AI_사용모델"] = ai_feedback.get("model_used", model)
                    result_record["AI_토큰수"] = ai_feedback.get("tokens_used", max_tokens)
                    result_record["AI_오류"] = ai_feedback.get("error", "")
                    
                    if ai_feedback.get("error"):
                        ai_fail_count += 1
                    else:
                        ai_success_count += 1
                else:
                    result_record["AI_장점"] = "AI 피드백이 비활성화되어 있습니다." if not enable_ai else "API 키가 제공되지 않았습니다."
                    result_record["AI_개선점"] = "AI 피드백이 비활성화되어 있습니다." if not enable_ai else "API 키가 제공되지 않았습니다."
                    result_record["AI_종합피드백"] = "키워드 기반 분석만 수행되었습니다."
                    result_record["AI_처리시간"] = 0
                    result_record["AI_사용모델"] = "none"
                    result_record["AI_토큰수"] = 0
                    result_record["AI_오류"] = ""
                
                result_record["분석시간"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                results.append(result_record)
                
                # 진행률 업데이트
                current_processed = len(results)
                progress = (current_processed + job_data["failed"]) / total_rows * 100
                store.update_job(job_id, {
                    "processed": current_processed,
                    "progress": min(progress, 100)
                })
                
                # AI 피드백 사용 시 속도 조절 (API 제한)
                if enable_ai and api_key:
                    await asyncio.sleep(1)  # 1초 대기
                else:
                    await asyncio.sleep(0.1)  # 0.1초 대기
                
            except Exception as e:
                logger.error(f"개별 분석 오류 - UID {uid}: {e}")
                current_failed = job_data["failed"] + 1
                store.update_job(job_id, {"failed": current_failed})
                continue
        
        # 결과 저장
        end_time = datetime.now()
        processing_time = end_time - job_data["start_time"]
        
        # 평균 점수 계산
        avg_score = 0
        if results:
            avg_score = sum(r["AIRISS_종합점수"] for r in results) / len(results)
        
        store.update_job(job_id, {
            "results": results,
            "status": "completed",
            "end_time": end_time,
            "processing_time": f"{processing_time.seconds}초",
            "average_score": round(avg_score, 1),
            "ai_success_count": ai_success_count,
            "ai_fail_count": ai_fail_count
        })
        
        # Excel 파일 생성
        if results:
            await create_excel_report(job_id, results, enable_ai)
        
        logger.info(f"분석 완료: {job_id}, 성공: {len(results)}, 실패: {job_data['failed']}, AI 성공: {ai_success_count}, AI 실패: {ai_fail_count}")
        
    except Exception as e:
        logger.error(f"분석 처리 오류: {e}")
        store.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })

async def create_excel_report(job_id: str, results: List[Dict], enable_ai: bool = False):
    """Excel 보고서 생성 - AI 피드백 포함 개선"""
    try:
        os.makedirs('results', exist_ok=True)
        
        # 결과 데이터프레임 생성
        df_results = pd.DataFrame(results)
        
        # 통계 요약 생성
        summary_stats = []
        
        # 전체 통계
        summary_stats.append({
            "항목": "전체 분석 건수",
            "값": len(results),
            "설명": "총 분석된 직원 수"
        })
        
        summary_stats.append({
            "항목": "평균 종합점수",
            "값": round(df_results["AIRISS_종합점수"].mean(), 1),
            "설명": "전체 직원 평균 점수"
        })
        
        # AI 피드백 통계 추가
        if enable_ai:
            ai_success = len(df_results[df_results["AI_오류"] == ""])
            ai_fail = len(df_results[df_results["AI_오류"] != ""])
            summary_stats.append({
                "항목": "AI 피드백 성공",
                "값": f"{ai_success}건",
                "설명": f"전체 {len(results)}건 중 AI 분석 성공"
            })
            if ai_fail > 0:
                summary_stats.append({
                    "항목": "AI 피드백 실패",
                    "값": f"{ai_fail}건",
                    "설명": "API 오류 등으로 AI 분석 실패"
                })
        
        summary_stats.append({
            "항목": "AI 피드백",
            "값": "활성화" if enable_ai else "비활성화",
            "설명": "OpenAI GPT 피드백 사용 여부"
        })
        
        # 등급별 분포
        grade_distribution = df_results["AIRISS_등급"].value_counts()
        for grade, count in grade_distribution.items():
            percentage = (count / len(results)) * 100
            summary_stats.append({
                "항목": f"{grade} 등급",
                "값": f"{count}명 ({percentage:.1f}%)",
                "설명": f"{grade} 등급 직원 수"
            })
        
        # 8대 영역별 평균 점수
        for dimension in AIRISS_FRAMEWORK.keys():
            col_name = f"{dimension}_점수"
            if col_name in df_results.columns:
                avg_score = round(df_results[col_name].mean(), 1)
                summary_stats.append({
                    "항목": f"{dimension} 평균",
                    "값": avg_score,
                    "설명": f"{dimension} 영역 평균 점수"
                })
        
        df_summary = pd.DataFrame(summary_stats)
        
        # Excel 파일 생성
        ai_suffix = "_AI완전분석" if enable_ai else "_키워드분석"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f'results/AIRISS{ai_suffix}_{timestamp}_{job_id[:8]}.xlsx'
        
        with pd.ExcelWriter(result_path, engine='openpyxl') as writer:
            # 메인 결과 시트
            df_results.to_excel(writer, index=False, sheet_name='AIRISS_분석결과')
            
            # 통계 요약 시트
            df_summary.to_excel(writer, index=False, sheet_name='통계요약')
            
            # 8대 영역별 상세 시트
            dimension_analysis = []
            for dimension in AIRISS_FRAMEWORK.keys():
                dimension_info = AIRISS_FRAMEWORK[dimension]
                col_name = f"{dimension}_점수"
                
                if col_name in df_results.columns:
                    scores = df_results[col_name]
                    dimension_analysis.append({
                        "영역": dimension,
                        "가중치": f"{dimension_info['weight']*100}%",
                        "설명": dimension_info['description'],
                        "평균점수": round(scores.mean(), 1),
                        "최고점수": round(scores.max(), 1),
                        "최저점수": round(scores.min(), 1),
                        "표준편차": round(scores.std(), 1),
                        "우수자수": len(scores[scores >= 80]),
                        "개선필요자수": len(scores[scores < 60])
                    })
            
            df_dimensions = pd.DataFrame(dimension_analysis)
            df_dimensions.to_excel(writer, index=False, sheet_name='영역별_분석')
            
            # AI 피드백 요약 시트 (AI 모드인 경우)
            if enable_ai and "AI_장점" in df_results.columns:
                ai_summary = []
                
                # AI 처리 시간 통계
                if "AI_처리시간" in df_results.columns:
                    avg_processing_time = df_results["AI_처리시간"].mean()
                    ai_summary.append({
                        "AI 분석 항목": "평균 처리시간",
                        "결과": f"{avg_processing_time:.2f}초",
                        "비고": "OpenAI API 응답 시간"
                    })
                
                # 피드백 품질 분석
                successful_feedback = len(df_results[df_results["AI_오류"] == ""])
                ai_summary.append({
                    "AI 분석 항목": "성공적 AI 분석",
                    "결과": f"{successful_feedback}/{len(results)}건",
                    "비고": f"성공률 {(successful_feedback/len(results)*100):.1f}%"
                })
                
                # 사용 모델 통계
                if "AI_사용모델" in df_results.columns:
                    model_used = df_results["AI_사용모델"].iloc[0] if len(df_results) > 0 else "unknown"
                    ai_summary.append({
                        "AI 분석 항목": "사용 모델",
                        "결과": model_used,
                        "비고": "OpenAI 모델"
                    })
                
                # 토큰 사용량 통계
                if "AI_토큰수" in df_results.columns:
                    total_tokens = df_results["AI_토큰수"].sum()
                    ai_summary.append({
                        "AI 분석 항목": "총 토큰 사용량",
                        "결과": f"{total_tokens:,} 토큰",
                        "비고": "OpenAI API 사용량"
                    })
                
                df_ai_summary = pd.DataFrame(ai_summary)
                df_ai_summary.to_excel(writer, index=False, sheet_name='AI_분석요약')
        
        # 작업 정보에 파일 경로 저장
        store.update_job(job_id, {"result_file": result_path})
        
        logger.info(f"Excel 보고서 생성 완료: {result_path} (AI 피드백: {enable_ai})")
        
    except Exception as e:
        logger.error(f"Excel 보고서 생성 오류: {e}")

@app.get("/status/{job_id}")
async def get_analysis_status(job_id: str):
    """분석 진행 상황 확인"""
    job_data = store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    # 처리 시간 계산
    if job_data["status"] == "completed" and "end_time" in job_data:
        processing_time = job_data["end_time"] - job_data["start_time"]
    else:
        processing_time = datetime.now() - job_data["start_time"]
    
    minutes = int(processing_time.total_seconds() // 60)
    seconds = int(processing_time.total_seconds() % 60)
    time_str = f"{minutes}분 {seconds}초" if minutes > 0 else f"{seconds}초"
    
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "total": job_data["total"],
        "processed": job_data["processed"],
        "failed": job_data["failed"],
        "progress": job_data["progress"],
        "processing_time": time_str,
        "average_score": job_data.get("average_score", 0),
        "error": job_data.get("error", ""),
        "ai_success_count": job_data.get("ai_success_count", 0),
        "ai_fail_count": job_data.get("ai_fail_count", 0)
    }

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """분석 결과 다운로드"""
    job_data = store.get_job(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="아직 완료되지 않은 작업입니다")
    
    result_file = job_data.get("result_file")
    if not result_file or not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="결과 파일을 찾을 수 없습니다")
    
    # 다운로드용 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ai_suffix = "AI완전분석" if job_data.get("enable_ai_feedback", False) else "키워드분석"
    filename = f"AIRISS_{ai_suffix}_{timestamp}.xlsx"
    
    return FileResponse(
        result_file,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy",
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "8_dimension_analysis": True,
            "ai_feedback_improved": True,
            "excel_export": True,
            "real_time_progress": True,
            "batch_processing": True,
            "openai_integration": analyzer.openai_available
        }
    }

# 메인 실행부
if __name__ == "__main__":
    # 필요한 디렉토리 생성
    os.makedirs('temp', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("🎯" + "="*60)
    print("🚀 AIRISS Enterprise AI Analysis System v3.1")
    print("="*64)
    print("✅ AI 피드백 기능 완전 개선")
    print("✅ OpenAI GPT-3.5/GPT-4 통합")
    print("✅ 8대 영역 정밀 분석")
    print("✅ 실시간 진행률 모니터링")
    print("✅ 완전한 Excel 보고서")
    print("="*64)
    print("🌐 브라우저에서 접속: http://localhost:8000")
    print("📊 API 문서: http://localhost:8000/docs")
    print("❤️  시스템 상태: http://localhost:8000/health")
    print("🤖 OpenAI 모듈:", "설치됨" if analyzer.openai_available else "미설치")
    print("="*64)
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,
            access_log=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ 서버 시작 오류: {e}")
        print("📋 해결 방법:")
        print("1. 포트 8000이 사용 중인지 확인")
        print("2. 관리자 권한으로 실행")
        print("3. 방화벽 설정 확인")
        print("4. OpenAI 모듈 설치: pip install openai")
        input("엔터를 눌러 종료...")