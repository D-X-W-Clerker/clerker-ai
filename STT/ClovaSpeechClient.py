## ClovaSpeechClient.py
## Clova Speech API 장문 인식 파일

import requests
import json

## Speaker Diarization
class ClovaSpeechClient:
    ## Clova Speech invoke URL
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/9022/1e7a0bb887f556370cd9ba0eaa2b96355035aec892732a47ce8c560721d8b8bc'
    ## Clova Speech secret key
    secret = '8c3b6f6ec8d449e9803600b33266a450'

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',  ## language
            'completion': completion,  ## sync / async
            'callback': callback,  ## url in async
            'userdata': userdata,  ## user information
            'wordAlignment': wordAlignment,  ## words with timestamp
            'fullText': fullText,  ## print full text
            'forbiddens': forbiddens,  ## distinguished words with commas
            'boostings': boostings,  ## specify important words / keywords
            'diarization': diarization,  ## speaker diarization
            'sed': sed, ## ambient sounds detection (clapping, background noise..)
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    ## object storage에 파일이 업로드 되어있는 경우
    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion, 
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    ## 파일을 업로드 해서 사용하는 경우
    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None, sed_enable=False):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,  ## async 방식 사용 시 저장할 url
            'userdata': userdata,  ## 사용자 정보, 프로젝트 이름 등 user 정보
            'wordAlignment': wordAlignment,  ## 발화와 timestamp 함께 반환
            'fullText': fullText,  ## 전체 텍스트 결과 출력
            'forbiddens': forbiddens,  ## 단어를 ','로 구분
            'boostings': boostings,  ## 중요한 단어 or 먼저 인식해야 하는 키워드 지정
            'diarization': diarization,  ## 화자 분리 기능
            'sed': {'enable': sed_enable} ## 박수소리, 배경소음 등 주변 소리 탐지 기능
            #'speakerCount' : 3 ## 발화자 몇 명인지 지정 가능
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }

        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response