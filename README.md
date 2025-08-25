# fullstack
🐢 Full-stack AI legal assistant app

## Shllter gradio 실행 방법
### 설치방법
#### Mac,Window Venv 기준
1. Github Pull 진행   
   1-1. clone  
    > git clone <repository_url>   

    1-2. 경로 이동   

    > cd <repository_name>
1. venv(가상환경) 설정   
    2-1. 설치 code   
    > python -m venv venv    

    2-2. 실행 code   
    >source venv/bin/activate  
1. PIP(패키지) 설정   
    3-1. pip 업그레이드   
    >python -m pip install --upgrade pip  

    3-2. pip 설치   
    > pip install gradio requests python-dotenv langchain-upstage langchain-core
1. Gradio 실행   
   > python shelter_gradio.py  
---
#### Mac,Window Conda 기준
1. Github Pull 진행   
   1-1. clone  
    > git clone <repository_url>   

    1-2. 경로 이동   

    > cd <repository_name>
1. venv(가상환경) 설정   
    2-1. 가상환경 생성  
    > conda create -n shelter_env python=3.10 -y

    2-2. 가상환경 실행
    >conda activate shelter_env.  
2. PIP(패키지) 설정   
    3-1. pip 업그레이드   
    >python -m pip install --upgrade pip  

    3-2. pip 설치   
    > pip install gradio requests python-dotenv langchain-upstage langchain-core
3. Gradio 실행   
   > python shelter_gradio.py  