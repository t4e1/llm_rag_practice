# Python 개발환경 설정 가이드
이 문서는 Python 개발 및 OpenAI API 사용을 위한 환경을 설정하는 단계별 가이드입니다.
---

## 목차
1. Miniforge 환경설정
2. VSCode 설치
3. Jupyter Notebook 설치
4. OpenAI API 설정
---

## 1. Miniforge 환경설정

### 1.1 Miniforge란?

Miniforge는 오픈소스 중심의 `conda-forge` 채널을 사용하는 경량화된 Python 패키지 및 환경 관리 도구입니다.

### 1.2 설치

아래의 링크에서 필요한 파일 다운로드

[Miniforge공식사이트](https://conda-forge.org/download/)


### 1.3 가상환경 생성 및 관리
```bash
# 새로운 가상환경 생성 (예: helloworld)
conda create -n agentworld python=3.12 -y

# 가상환경 목록 확인
conda env list

# 가상환경 활성화
conda activate agentworld

# pip으로 설치
pip install jupyter notebook

# 가상환경 비활성화
conda deactivate
```

---
## 2. VSCode 설치
### 2.1 설치
[공식 사이트](https://code.visualstudio.com/)에서 설치 가능합니다.

### 2.2 필수 확장(Extensions) 설치
- python
- jupyter
---

## 3. API Key 설정

### 4.1 OpenAI 계정 생성 및 API 키 발급
1. https://platform.openai.com 에 접속하여 계정을 생성합니다.
2. 로그인 후 API Keys 섹션으로 이동합니다.
3. "Create new secret key"를 클릭하여 새로운 API 키를 생성합니다.
4. 생성된 API 키를 안전한 곳에 복사하여 보관합니다.

### 4.2 환경 변수 설정
API 키를 환경 변수로 설정하여 코드에서 안전하게 사용할 수 있습니다.


### 4.2 LangSmith 계정 생성 및 API Key 발급
1. https://smith.langchain.com 에 접속하여 계정을 생성합니다.
3. 로그인 후 우측 상단의 설정(Settings) 메뉴로 이동합니다.
4. "API Keys" 탭을 클릭합니다.
5. "Create API Key"를 클릭하여 새로운 API 키를 생성합니다.