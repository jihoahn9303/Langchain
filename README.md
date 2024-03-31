# Making Langchain Application

**Langchain 프레임워크를 이용하여 LLM을 학습하고 애플리케이션을 만들어가는 공간입니다.**

생각나는 애플리케이션을 작게나마 지속적으로 만들고 업데이트 할 예정입니다!


## Smart Document Retriever

RAG(Retrieval Augmented Generation) 방법을 활용하여, 주어진 문서에 대해 세부 사항을 파악해주는 똑똑한 Chatbot을 만들었습니다!

깔끔한 웹 인터페이스를 제작하기 위하여, Streamlit으로 애플리케이션을 제작하였습니다 :)

이때, LLM Chatbot에 필요한 문서 호출 시간을 짧게하기 위하여, Langchain의 Stuff를 커스텀 체인으로 구현했습니다.

또한, Streamlit의 dataflow로 인한 리소스(LLM, Memory etc..) 호출 비용을 줄이기 위해, 캐싱 기능을 사용했습니다.

아래에 데모 영상을 첨부했어요🦾
